import argparse
from pathlib import Path
from random import randint

import cv2

from super_gradients.training.utils.visualization.detection import draw_bbox
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare results based on images from directory.")

    parser.add_argument(
        "--data_dir",
        default=Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic"),  # todo remove
        type=Path,
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--images_dir",
        default=Path("PNG"),
        type=Path,
        help="Path to directory with input images.",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic/SANITY_CHECK"),  # todo remove
        type=Path,
        help="Path to directory where results should be saved.",
    )
    parser.add_argument(
        "--split_info_pth",
        default="COCO/test.json",
        type=Path,
        help="Path to COCO output json annotation file.",
    )
    return parser.parse_args()


def main(
    data_dir,
    images_dir,
    split_info_pth,
    output_dir,
) -> None:
    dataset = COCOFormatDetectionDataset(
        data_dir=data_dir,
        json_annotation_file=split_info_pth,
        images_dir=images_dir,
    )

    colours = [(randint(0, 255), randint(0, 255), randint(0, 255)) for cl in dataset.classes]

    for no, sample in enumerate(dataset):
        image, target, _ = sample

        for box in target:
            xmin, ymin, xmax, ymax, class_id = box
            class_name = dataset.classes[int(class_id)]
            colour = colours[int(class_id)]
            box_thickness = None

            image = draw_bbox(image, class_name, colour, box_thickness, x1=int(xmin), y1=int(ymin), x2=int(xmax), y2=int(ymax))

        output_path = str(output_dir / f"sample_{no}.png")
        cv2.imwrite(output_path, image)
        print(f"Saved in: {output_path}.")

    print("Processing done.")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.data_dir,
        args.images_dir,
        args.split_info_pth,
        args.output_dir,
    )
