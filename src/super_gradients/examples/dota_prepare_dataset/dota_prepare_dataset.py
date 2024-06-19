"""
This script slices the DOTA dataset into tiles of a usable size for training a model.
The tiles are saved in the output directory with the same structure as the input directory.

To use this script you should download the DOTA dataset from the official website:
https://captain-whu.github.io/DOTA/dataset.html

The dataset should be organized as follows:
    dota
    └── train
        ├── images
        │   ├─ 000000000001.jpg
        │   └─ ...
        └── ann
            └─ 000000000001.txt
    └── val
        ├── images
        │   ├─ 000000000002.jpg
        │   └─ ...
        └── ann
            └─ 000000000002.txt

Example usage:
    python dota_prepare_dataset.py --input_dir /path/to/dota --output_dir /path/to/dota-sliced

After running this script you can use  /path/to/dota-sliced as the data_dir argument for training a model on DOTA dataset.
"""

import argparse
from pathlib import Path

import cv2
from super_gradients.training.datasets import DOTAOBBDataset


def main():
    parser = argparse.ArgumentParser(description="Slice DOTA dataset into tiles of usable size for training a model")
    parser.add_argument("--input_dir", help="Where the full coco dataset is stored", required=True)
    parser.add_argument("--output_dir", help="Where the resulting data should be stored", required=True)
    parser.add_argument("--ann_subdir_name", default="ann", help="Name of the annotations subdirectory")
    parser.add_argument("--output_ann_subdir_name", default="ann-obb", help="Name of the output annotations subdirectory")
    parser.add_argument("--num_workers", default=cv2.getNumberOfCPUs() // 2)
    args = parser.parse_args()

    cv2.setNumThreads(cv2.getNumberOfCPUs() // 4)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ann_subdir_name = str(args.ann_subdir_name)
    output_ann_subdir_name = str(args.output_ann_subdir_name)
    DOTAOBBDataset.slice_dataset_into_tiles(
        data_dir=input_dir / "train",
        output_dir=output_dir / "train",
        input_ann_subdir_name=ann_subdir_name,
        output_ann_subdir_name=output_ann_subdir_name,
        tile_size=1024,
        tile_step=512,
        scale_factors=(0.75, 1, 1.25),
        min_visibility=0.4,
        min_area=8,
        num_workers=args.num_workers,
    )

    DOTAOBBDataset.slice_dataset_into_tiles(
        data_dir=input_dir / "val",
        output_dir=output_dir / "val",
        input_ann_subdir_name=ann_subdir_name,
        output_ann_subdir_name=output_ann_subdir_name,
        tile_size=1024,
        tile_step=1024,
        scale_factors=(1,),
        min_visibility=0.4,
        min_area=8,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
