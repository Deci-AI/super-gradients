import os
from collections import defaultdict

import numpy as np
import torch
from fire import Fire
from super_gradients.training import models
from super_gradients.training.processing.defaults import default_yolo_nas_r_dota_processing_params
from tqdm import tqdm
import cv2
import PIL


@torch.no_grad()
@torch.jit.optimized_execution(False)
def main(
    model_name,
    checkpoint_path,
    images_dir,
    submission_dir=None,
    device="cpu",
    min_confidence=0.1,
):
    PIL.Image.MAX_IMAGE_PIXELS = None

    checkpoint_path = os.path.expanduser(checkpoint_path)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)

    images_dir = os.path.expanduser(images_dir)
    if not os.path.isabs(images_dir):
        images_dir = os.path.abspath(images_dir)

    if submission_dir is None:
        submission_dir = os.path.join(os.path.dirname(checkpoint_path), "submission")

    print(f"checkpoint_path: {checkpoint_path}")
    print(f"model_name:      {model_name}")
    print(f"images_dir:      {images_dir}")
    print(f"device:          {device}")
    print(f"min_confidence:  {min_confidence}")
    print(f"submission_dir:  {submission_dir}")

    # Load model
    model = models.get(model_name, checkpoint_path=checkpoint_path, num_classes=18)
    model.set_dataset_processing_params(**default_yolo_nas_r_dota_processing_params())
    model = model.to(device).eval()
    model.prep_model_for_conversion(input_size=(1024, 1024))

    pipeline = model._get_pipeline(
        fuse_model=False, skip_image_resizing=True, iou=0.6, pre_nms_max_predictions=32768, conf=min_confidence, post_nms_max_predictions=4096
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
    # order images by filesize (largest first)
    # If inference on the largest image works, it should work on the smaller ones as well
    images_list = list(sorted(images_list, key=lambda x: os.path.getsize(x), reverse=True))
    # images_list = list(sorted(images_list, key=lambda x: os.path.getsize(x), reverse=False))

    # os.makedirs(model_name, exist_ok=True)
    visualizations_dir = os.path.join(submission_dir, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)

    for image_path in tqdm(images_list, desc="Predicting & Saving results"):
        image_name = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_name)[0]
        predictions_result = pipeline(image_path)
        # predictions_result.save(os.path.join(visualizations_dir, image_name))
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
    Fire(main)
