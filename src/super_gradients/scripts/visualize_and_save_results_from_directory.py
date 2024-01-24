import argparse
import json
from pathlib import Path

import cv2

from super_gradients.training import models
from super_gradients.training.transforms.utils import _rescale_and_pad_to_size


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
        default=str(
            "/mnt/ml-team/homes/marianna.parzych/Unstructured/super-gradients-fork/checkpoints/training_512x512/RUN_20240120_141334_742570/ckpt_best.pth"
        ),  # todo remove
        type=str,
        help="Path to model checkpoint to use.",
    )
    parser.add_argument(
        "--input_dir",
        default=Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic/PNG"),  # todo remove
        type=Path,
        help="Path to directory with input images.",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic/RESULTS"),  # todo remove
        type=Path,
        help="Path to directory where results should be saved.",
    )
    parser.add_argument(
        "--split_info_pth",
        type=Path,
        help="Path to COCO output json annotation file.",
    )
    return parser.parse_args()


def main(
    model_type,
    num_classes,
    checkpoint_path,
    input_dir,
    output_dir,
    split_info_pth,
    size=(1024, 1024),  # 512x512, 1024x1024, 1664x1664
) -> None:
    model = models.get(model_type, num_classes=num_classes, checkpoint_path=checkpoint_path)

    if split_info_pth is not None:
        with open(split_info_pth, "r") as file:
            split_info = json.load(file)
        images = [image["file_name"] for image in split_info["images"]]
    else:
        images = [image.name for image in input_dir.iterdir()]

    for image_name in images:
        image_path = str(input_dir / image_name)

        image_array = cv2.imread(image_path)
        resized_image, r = _rescale_and_pad_to_size(image_array, size)

        output = model.predict(resized_image)
        output_image = output.draw()
        output_path = str(output_dir / image_name)
        cv2.imwrite(output_path, output_image)
        print(f"Saved in: {output_path}.")

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
    )
