# This example script shows how to use a trained YoloNAS-R model to make predictions for DOTA 2.0 test-dev
# dataset and save the results in the format required for the DOTA 2.0 test-dev submission.
# Prediction script does not use tiled inference and process entire image at once.
# Since some images in the DOTA dataset are very large, it is recommended to use a machine with a lot of RAM (128+Gb)
# to run this script. A 24Gb GPU is enough to fit the largest images in the DOTA dataset,so we use CPU for inference.
#
# Example usage:
# dota_predict_test_dev.py yolo_nas_r_s checkpoints/yolo_nas_r_s_dota2_best.pth /path/to/DOTA-v2.0/test-dev/images /path/to/save/submission
from pathlib import Path

import cv2
import PIL
import os
import argparse
from collections import defaultdict

import numpy as np
import torch
from super_gradients.training import models
from tqdm import tqdm


@torch.no_grad()
@torch.jit.optimized_execution(False)
def main():
    args = argparse.ArgumentParser()
    args.add_argument("model_name", type=str, default=None, required=True, help="Model name")
    args.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint")
    args.add_argument("images_dir", type=str, help="Path to the images directory with DOTA test-dev images")
    args.add_argument("--submission_dir", type=str, default=None, help="Path to save submission files")
    args.add_argument("--visualization_dir", type=str, default=None, help="Path to save visualizations")
    args.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    args.add_argument("--min_confidence", type=float, default=0.1, help="Minimum confidence threshold")
    args.add_argument("--iou_threshold", type=float, default=0.2, help="IoU threshold for NMS")
    args = args.parse_args()

    model_name = args.model_name
    checkpoint_path = args.checkpoint_path
    images_dir = args.images_dir
    submission_dir = args.submission_dir
    visualization_dir = args.visualization_dir
    device = args.device
    min_confidence = args.min_confidence
    iou_threshold = args.iou_threshold

    PIL.Image.MAX_IMAGE_PIXELS = None

    checkpoint_path = os.path.expanduser(checkpoint_path)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)

    images_dir = os.path.expanduser(images_dir)
    if not os.path.isabs(images_dir):
        images_dir = os.path.abspath(images_dir)

    if submission_dir is None:
        submission_dir = os.path.join(os.path.dirname(checkpoint_path), str(Path(checkpoint_path).stem) + "_dota_submission")

    print(f"checkpoint_path: {checkpoint_path}")
    print(f"model_name:      {model_name}")
    print(f"images_dir:      {images_dir}")
    print(f"device:          {device}")
    print(f"min_confidence:  {min_confidence}")
    print(f"iou_threshold:   {iou_threshold}")
    print(f"submission_dir:  {submission_dir}")

    # Load model
    model = models.get(model_name, checkpoint_path=checkpoint_path, num_classes=18)
    model = model.to(device).eval()
    model.prep_model_for_conversion(input_size=(1024, 1024))

    pipeline = model._get_pipeline(
        fuse_model=False, skip_image_resizing=True, iou=iou_threshold, pre_nms_max_predictions=32768, conf=min_confidence, post_nms_max_predictions=4096
    )
    class_names = pipeline.class_names

    model = torch.jit.trace(
        model,
        example_inputs=torch.randn(1, 3, 1024, 1024).to(device),
    )
    pipeline.model = model

    all_detections = defaultdict(list)

    images_list = os.listdir(images_dir)  # [:5]
    images_list = [os.path.join(images_dir, image_name) for image_name in images_list]
    # Order images by filesize (largest first)
    # If inference on the largest image works, it should work on the smaller ones as well
    images_list = list(sorted(images_list, key=lambda x: os.path.getsize(x), reverse=True))

    if visualization_dir is not None:
        os.makedirs(visualization_dir, exist_ok=True)

    for image_path in tqdm(images_list, desc="Predicting & Saving results"):
        image_name = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_name)[0]
        predictions_result = pipeline(image_path)
        if visualization_dir is not None:
            predictions_result.save(os.path.join(visualization_dir, image_name))
        data = predictions_result.prediction

        print(f"Predictions for {image_name} - {len(data.labels)} objects")

        for class_label, score, (cx, cy, w, h, r) in zip(data.labels, data.confidence, data.rboxes_cxcywhr):
            class_name = class_names[int(class_label)]
            if not all(np.isfinite(value) for value in [score, cx, cy, w, h, r]):
                print(f"Skipping prediction for {image_name} because of invalid values: {class_name} {score}, {cx}, {cy}, {w}, {h} {r}")

            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cv2.boxPoints(((cx, cy), (w, h), np.rad2deg(r)))

            prediction_line = f"{image_name_no_ext} {score:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {x3:.2f} {y3:.2f} {x4:.2f} {y4:.2f}\n"
            all_detections[class_name].append(prediction_line)

    os.makedirs(submission_dir, exist_ok=True)

    for class_name, detections in all_detections.items():
        with open(f"{submission_dir}/Task1_{class_name}.txt", "w") as f:
            f.writelines(detections)


if __name__ == "__main__":
    main()
