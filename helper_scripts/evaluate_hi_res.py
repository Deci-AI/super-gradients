"""
This script is used to evaluate the hi-res model outputs using the labels provided in the labels_path.

hi-res model outputs can be downloaded from S3:
s3://utic-comparative-metrics/mini-holistic/unstructured-core-hi_res/

Labels can be downloaded from DVC:
dvc-data-registry/holistic-mini-pdf-image-dataset/mini-holistic-all/ls/export_45956_project-45956-at-2024-01-10-23-16-24cfbda6.json

To run the script, use the following command:
python helper_scripts/evaluate_hi_res.py --output_dir <path_to_model_outputs> --labels_path <path_to_labels>
"""


import argparse
import dotenv
import json
import logging
import os
from pathlib import Path

import neptune
import numpy as np
import torch

from super_gradients.training import utils as core_utils
from super_gradients.training.metrics.detection_metrics import DetectionMetrics, DetectionMetrics_09
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from mappings import HI_RES_ELEMENT_TYPES
from utils import IdentityPostPredictionCallback

dotenv.load_dotenv()
logger = logging.getLogger()

# Some predictions do not have a detection class probability. This value is used as a default when the probability is
# not found. This value should be set to 1.0 since hi-res model outputs are not filtered by detection class probability.
DEFAULT_DETECTION_CLASS_PROB = 0.2

NEPTUNE_PARAMS = {
    "name": "hi_res_evaluation",
    "tags": ["hi_res", "eval", "CORE-3977"],
}

METRICS = [
    DetectionMetrics(
        num_cls=23,
        post_prediction_callback=IdentityPostPredictionCallback(),
        normalize_targets=True,
        include_classwise_f1=True,
        include_classwise_ap=True,
        include_classwise_precision=True,
        include_classwise_recall=True,
        class_names=[cat['name'] for cat in HI_RES_ELEMENT_TYPES]),
    DetectionMetrics_09(
        num_cls=23,
        post_prediction_callback=IdentityPostPredictionCallback(),
        normalize_targets=True,
        include_classwise_f1=True,
        include_classwise_ap=True,
        include_classwise_precision=True,
        include_classwise_recall=True,
        class_names=[cat['name'] for cat in HI_RES_ELEMENT_TYPES])
]

MINIHOLISTIC_IDS = {cat['name']: cat['id'] for cat in HI_RES_ELEMENT_TYPES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset to standard COCO format.")

    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to directory json model outputs are stored.",
    )
    parser.add_argument(
        "--labels_path",
        type=Path,
        help="Path to json file containing labels for the documents in the output directory",
    )
    parser.add_argument(
        "--default_detection_class_prob",
        type=float,
        default=DEFAULT_DETECTION_CLASS_PROB,
        help="Default detection class probability to use when the probability is not found in the predictions",
    )

    return parser.parse_args()


def format_predictions(predictions):
    """
    Format predictions to a torch tensor with shape (N, 6), where N is the number of predictions in the batch.
    """
    formatted_predictions = np.zeros((len(predictions), 6))
    for i, pred in enumerate(predictions):
        formatted_predictions[i, 0] = pred["metadata"]["coordinates"]["points"][0][0]
        formatted_predictions[i, 1] = pred["metadata"]["coordinates"]["points"][0][1]
        formatted_predictions[i, 2] = pred["metadata"]["coordinates"]["points"][2][0]
        formatted_predictions[i, 3] = pred["metadata"]["coordinates"]["points"][2][1]
        try:
            formatted_predictions[i, 4] = pred["metadata"]["detection_class_prob"]
        except KeyError:
            formatted_predictions[i, 4] = DEFAULT_DETECTION_CLASS_PROB
            logger.warning(f"Detection class probability not found for {pred['element_id']}")
        formatted_predictions[i, 5] = MINIHOLISTIC_IDS[pred["type"]]
    return torch.from_numpy(formatted_predictions)


def format_labels(labels):
    """
    Format labels to a torch tensor with shape (N, 6), where N is the number of annotations in the labels.
    """
    # labels should contain single page information
    assert len(labels) == 1

    formatted_labels = np.empty(shape=(0, 6))

    for page_i, page_labels in enumerate(labels):
        for annotation in page_labels["result"]:
            if annotation["type"] == "labels":
                category = annotation["value"]["labels"][0]
                x0 = annotation["value"]["x"] / 100 * annotation["original_width"]
                y0 = annotation["value"]["y"] / 100 * annotation["original_height"]
                width = annotation["value"]["width"] / 100 * annotation["original_width"]
                height = annotation["value"]["height"] / 100 * annotation["original_height"]
                xyxy = np.array([[x0, y0, x0 + width, y0 + height]])
                cx, cy, w, h = core_utils.detection_utils.xyxy2cxcywh(xyxy)[0]
                category_id = MINIHOLISTIC_IDS[category]
                labels_row = np.array([[page_i, category_id, cx, cy, w, h]])

                formatted_labels = np.concatenate([formatted_labels, labels_row])

    return torch.from_numpy(formatted_labels)


def get_width_height(page_predictions, page_labels, document_name, page_number):
    """
    Get the width and height of the image and label for a given page. If the dimensions do not match, log a warning.
    """
    try:
        img_width = page_predictions[0]["metadata"]["coordinates"]["layout_width"]
        img_height = page_predictions[0]["metadata"]["coordinates"]["layout_height"]
    except IndexError:
        logger.info(f"No predictions found for document {document_name}; page {page_number}")
        img_width = None
        img_height = None
    try:
        label_width = page_labels[0]["result"][0]["original_width"]
        label_height = page_labels[0]["result"][0]["original_height"]
    except IndexError:
        logger.info(f"No labels found for document {document_name}; page {page_number}")
        label_width = None
        label_height = None

    # if the image and label dimensions do not match, log a warning
    if img_width != label_width or img_height != label_height:
        if img_width and label_width and img_height and label_height:
            logger.warning(f"Image and label dimensions do not match for document {document_name}; "
                           f"page {page_number}. "
                           f"Image: {img_width}x{img_height}, Label: {label_width}x{label_height}")
            return None, None

    width = img_width if img_width else label_width or 0
    height = img_height if img_height else label_height or 0

    return width, height


def main(
    output_dir: Path,
    labels_path: Path
):
    neptune_run = neptune.init_run(**NEPTUNE_PARAMS)

    device = "cpu"

    with open(labels_path, 'r') as file:
        logger.info(f"Loading labels from {labels_path}")
        labels = json.load(file)

    document_labels = {}

    for document_filename in os.listdir(output_dir):
        logger.info(f"Processing {document_filename}")
        document_name = document_filename.split('.')[0]
        document_labels[document_name] = {}

        if document_filename.endswith('.json'):
            with open(output_dir / document_filename, 'r') as file:
                logger.info(f"Loading predictions from {document_filename}")
                document_predictions = json.load(file)

            for label in labels:
                if document_name in label['file_upload']:
                    page_number = os.path.splitext(label['file_upload'])[0].split('_')[-1]
                    try:
                        page_number = int(page_number)
                    except ValueError:
                        page_number = 0
                    document_labels[document_name][page_number] = label['annotations']

            for page_number, page_labels in document_labels[document_name].items():
                # labels are iterated from 0
                page_predictions = [p for p in document_predictions if p['metadata']['page_number'] == page_number + 1]
                formatted_predictions = format_predictions(page_predictions)
                formatted_labels = format_labels(page_labels)

                width, height = get_width_height(page_predictions, page_labels, document_name, page_number)
                if width is None or height is None:
                    logger.warning(f"Skipping page {page_number} of document {document_name} "
                                   f"due to mismatched dimensions")
                    continue

                # imgs information is not used in the metric; only height and width are used
                imgs = torch.zeros((1, 3, height, width))

                for metric in METRICS:
                    metric.update(formatted_predictions, formatted_labels, device=device, inputs=imgs)

    metric_outputs = []
    for metric in METRICS:
        results = metric.compute()
        metric_outputs.append(results)
        for k, v in results.items():
            neptune_run[k] = v

    logger.info(f"Metric outputs: {metric_outputs}")

    neptune_run.stop()
    return metric_outputs


if __name__ == "__main__":
    args = parse_args()
    main(args.output_dir, args.labels_path)
