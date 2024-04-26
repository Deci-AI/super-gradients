import argparse
from pathlib import Path

import cv2
from super_gradients.training.datasets import DOTAOBBDataset


def main():
    parser = argparse.ArgumentParser(description="Slice DOTA dataset into tiles of usable size for training a model")
    parser.add_argument("--input_dir", help="Where the full coco dataset is stored", required=True)
    parser.add_argument("--output_dir", help="Where the resulting data should be stored", required=True)
    parser.add_argument("--num_workers", default=cv2.getNumberOfCPUs() // 2)
    args = parser.parse_args()
    ann_subdir_name = "ann-obb"

    cv2.setNumThreads(cv2.getNumberOfCPUs() // 4)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    DOTAOBBDataset.slice_dataset_into_tiles(
        data_dir=input_dir / "train",
        output_dir=output_dir / "train",
        ann_subdir_name=ann_subdir_name,
        tile_size=1024,
        tile_step=512,
        scale_factors=(0.75, 1, 1.25),
        min_visibility=0.5,
        min_area=8,
        num_workers=args.num_workers,
    )

    DOTAOBBDataset.slice_dataset_into_tiles(
        data_dir=input_dir / "val",
        output_dir=output_dir / "val",
        ann_subdir_name=ann_subdir_name,
        tile_size=1024,
        tile_step=1024,
        scale_factors=(1,),
        min_visibility=0.5,
        min_area=8,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
