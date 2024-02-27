"""
This is a helper scripts that performes inference on images from given directory and saves detection visualizations.

>> python src/super_gradients/scripts/visualize_and_save_results_from_directory.py --model_type yolox_l --num_classes 11 --checkpoint_path \
    /path/to/model/checkpoint.pth --input_dir /path/to/directory/with/input/images/ --output_dir /path/to/directory/where/results/should/be/saved
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path

from super_gradients.training import models
from super_gradients.training.transforms.utils import _rescale_and_pad_to_size, _rescale_xyxy_bboxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare results based on images from directory.")

    parser.add_argument(
        "--model_type",
        default="yolox_l",
        type=str,
        help="Model type to use.",
    )
    parser.add_argument(
        "--num_classes",
        default=11,
        type=int,
        help="Number of classes.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to model checkpoint to use.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Path to directory with input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to directory where results should be saved.",
    )
    parser.add_argument(
        "--split_info_pth",
        type=Path,
        help="Path to COCO output json annotation file. If None, all images in directory will be used.",
    )
    parser.add_argument(
        "--iou",
        default=0.3,
        type=float,
        help="IoU threshold for the non-maximum suppression (NMS) algorithm.",
    )
    parser.add_argument(
        "--conf",
        default=0.3,
        type=float,
        help="Confidence threshold. Predictions below this threshold are discarded.",
    )
    parser.add_argument(
        "--res",
        type=int,
        help="Resize the image to resolution before inference.",
    )
    parser.add_argument(
        "--output_json",
        default=False,
        type=bool,
        help="If True, saves results in json format.",
    )
    return parser.parse_args()


def main(
    model_type: str,
    num_classes: int,
    checkpoint_path: Path,
    input_dir: Path,
    output_dir: Path,
    split_info_pth: Path,
    iou: float,
    conf: float,
    res: int | None,
    output_json: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = models.get(model_type, num_classes=num_classes, checkpoint_path=checkpoint_path)

    if split_info_pth is not None:
        with open(split_info_pth, "r") as file:
            split_info = json.load(file)
        files = [image["file_name"] for image in split_info["images"]]
    else:
        files = [file.name for file in input_dir.iterdir()]

    for file_name in files:
        file_path = input_dir / file_name

        page_count = 0
        if file_path.suffix.lower() == ".pdf":
            images = convert_from_path(file_path)
            images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in images]
            if len(images) < 1:
                print(f"Found 0 images in file: {file_path}")
                continue
        else:
            images = [cv2.imread(str(file_path), cv2.IMREAD_COLOR)]  # DetectionDataset loads images as BGR, and channel last (HWC).
            if images[0] is None:
                print(f"Couldn't load image: {file_path}")
                continue

        for image_array in images:
            if res:
                image_input, r = _rescale_and_pad_to_size(image_array, (res, res))  # yolox only resizes images, no other transforms
                # todo: support for other backbones
            else:
                image_input = image_array
                r = None

            output = model.predict(image_input, iou=iou, conf=conf)

            if r:
                output.image = image_array
                bboxes = output.prediction.bboxes_xyxy
                output.prediction.bboxes_xyxy = _rescale_xyxy_bboxes(bboxes, 1 / r)

            image_output = output.draw()

            suffix = Path(file_name).suffix
            output_path = str(output_dir / f"{file_name.removesuffix(suffix)}_{page_count}.png")
            cv2.imwrite(output_path, image_output)
            print(f"Saved in: {output_path}.")

            if output_json:
                dict_output = {
                    "class_names": output.class_names,
                    "bboxes_xyxy": output.prediction.bboxes_xyxy.tolist(),
                    "confidence": output.prediction.confidence.tolist(),
                    "labels": output.prediction.labels.tolist(),
                    "image_shape": output.prediction.image_shape,
                }
                out_json_path = output_path.with_suffix(".json")
                with open(out_json_path, "w") as file:
                    json.dump(dict_output, file)

                print(f"Saved in: {out_json_path}")

    print("Processing done.")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.model_type,
        args.num_classes,
        args.checkpoint_path,
        args.input_dir,
        args.output_dir,
        args.split_info_pth,
        args.iou,
        args.conf,
        args.res,
        args.output_json,
    )
