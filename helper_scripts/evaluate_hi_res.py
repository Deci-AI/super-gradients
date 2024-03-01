"""
This script is used to evaluate the hi-res model outputs using the labels provided in the labels_path.

hi-res model outputs can be downloaded from S3:
s3://utic-comparative-metrics/mini-holistic/unstructured-core-hi_res/

Labels can be downloaded from DVC:
dvc-data-registry/holistic-mini-pdf-image-dataset/mini-holistic-all/ls/export_45956_project-45956-at-2024-01-10-23-16-24cfbda6.json

To run the script, use the following command:
python helper_scripts/evaluate_hi_res.py --output_dir <path_to_model_outputs> --labels_path <path_to_labels> [--default_detection_class_prob 1.0]
"""

import argparse
import logging
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import dotenv
import json
import neptune
import numpy as np
import torch

from super_gradients.training.utils import detection_utils as core_utils
from super_gradients.training.metrics.detection_metrics import DetectionMetrics, DetectionMetrics_09
from mappings import HI_RES_ELEMENT_TYPES
from utils import IdentityPostPredictionCallback

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DETECTION_CLASS_PROB = 1.0

NEPTUNE_PARAMS = {
    "name": "hi_res_evaluation",
    "tags": ["hi_res", "eval", "CORE-3977"],
}

MINIHOLISTIC_IDS = {cat["name"]: cat["id"] for cat in HI_RES_ELEMENT_TYPES}

METRICS = [
    DetectionMetrics(
        num_cls=23,
        post_prediction_callback=IdentityPostPredictionCallback(),
        normalize_targets=True,
        include_classwise_f1=True,
        include_classwise_ap=True,
        include_classwise_precision=True,
        include_classwise_recall=True,
        class_names=[cat["name"] for cat in HI_RES_ELEMENT_TYPES],
    ),
    DetectionMetrics_09(
        num_cls=23,
        post_prediction_callback=IdentityPostPredictionCallback(),
        normalize_targets=True,
        include_classwise_f1=True,
        include_classwise_ap=True,
        include_classwise_precision=True,
        include_classwise_recall=True,
        class_names=[cat["name"] for cat in HI_RES_ELEMENT_TYPES],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hi-res model outputs.")

    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory where JSON model outputs are stored.")
    parser.add_argument(
        "--labels_path", type=Path, required=True,
        help="JSON file containing labels for the documents.")
    parser.add_argument(
        "--default_detection_class_prob", type=float, default=DEFAULT_DETECTION_CLASS_PROB,
        help="Default detection class probability when not found."
    )

    return parser.parse_args()


def format_predictions(predictions: List[Dict], default_prob: float) -> torch.Tensor:
    preds = np.zeros((len(predictions), 6))
    for i, pred in enumerate(predictions):
        coords = pred["metadata"]["coordinates"]["points"]
        preds[i, :4] = coords[0][0], coords[0][1], coords[2][0], coords[2][1]
        preds[i, 4] = pred.get("metadata", {}).get("detection_class_prob", default_prob)
        preds[i, 5] = MINIHOLISTIC_IDS.get(pred["type"])
    return torch.tensor(preds, dtype=torch.float32)


def format_labels(labels: List[Dict]) -> torch.Tensor:
    assert len(labels) == 1, "Labels should contain single page information"
    annotations = labels[0]["result"]
    formatted_labels = [format_single_annotation(anno) for anno in annotations if anno["type"] == "labels"]
    if not formatted_labels:
        return torch.tensor(np.empty(shape=(0, 6)), dtype=torch.float32)
    return torch.tensor(formatted_labels, dtype=torch.float32)


def format_single_annotation(annotation: Dict) -> List[float]:
    category = annotation["value"]["labels"][0]
    x0, y0 = annotation["value"]["x"] / 100 * annotation["original_width"], annotation["value"]["y"] / 100 * annotation["original_height"]
    width, height = annotation["value"]["width"] / 100 * annotation["original_width"], annotation["value"]["height"] / 100 * annotation["original_height"]
    xyxy = np.array([[x0, y0, x0 + width, y0 + height]])
    cx, cy, w, h = core_utils.xyxy2cxcywh(xyxy)[0]
    category_id = MINIHOLISTIC_IDS[category]
    return [0, category_id, cx, cy, w, h]


def get_width_height(page_predictions: List[Dict], page_labels: List[Dict], document_name: str, page_number: int) -> Tuple[Optional[int], Optional[int]]:
    try:
        img_dimensions = page_predictions[0]["metadata"]["coordinates"]
        img_width, img_height = img_dimensions["layout_width"], img_dimensions["layout_height"]
    except IndexError:
        logger.info(f"No predictions found for {document_name}, page {page_number}")
        img_width, img_height = None, None

    try:
        label_dims = page_labels[0]["result"][0]
        label_width, label_height = label_dims["original_width"], label_dims["original_height"]
    except IndexError:
        logger.info(f"No labels found for {document_name}, page {page_number}")
        label_width, label_height = None, None

    if img_width != label_width or img_height != label_height:
        if img_width and label_width and img_height and label_height:
            logger.warning(f"Document {document_name}, page {page_number}: Dimension mismatch.")
            return None, None

    return img_width or label_width or 0, img_height or label_height or 0


def main(output_dir: Path, labels_path: Path, default_detection_class_prob: float):
    neptune_run = neptune.init_run(**NEPTUNE_PARAMS)
    logger.info(f"Neptune run created: {neptune_run}")

    device = "cpu"

    with open(labels_path, "r") as file:
        logger.info(f"Loading labels from {labels_path}")
        labels = json.load(file)

    document_labels = {}

    for document_filename in os.listdir(output_dir):
        logger.info(f"Processing {document_filename}")
        document_name = Path(document_filename).stem
        document_labels[document_name] = {}

        if document_filename.endswith(".json"):
            with open(output_dir / document_filename, "r") as file:
                logger.info(f"Loading predictions from {document_filename}")
                document_predictions = json.load(file)

            for label in labels:
                if document_name in label["file_upload"]:
                    page_number = Path(label["file_upload"]).stem.split("_")[-1]
                    try:
                        page_number = int(page_number)
                    except ValueError:
                        page_number = 0
                    document_labels[document_name][page_number] = label["annotations"]

            for page_number, page_labels in document_labels[document_name].items():
                # labels are iterated from 0
                page_predictions = [p for p in document_predictions if p["metadata"]["page_number"] == page_number + 1]
                formatted_predictions = format_predictions(page_predictions, default_detection_class_prob)
                formatted_labels = format_labels(page_labels)

                width, height = get_width_height(page_predictions, page_labels, document_name, page_number)
                if width is None or height is None:
                    logger.warning(f"Skipping page {page_number} of document {document_name} " f"due to mismatched dimensions")
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
    main(args.output_dir, args.labels_path, args.default_detection_class_prob)
