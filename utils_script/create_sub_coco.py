from pathlib import Path
import json
import shutil
import argparse


def copy_train_val_to_new_dir(input_data_dir: str, dest_data_dir: str):
    """Create a subset of Coco2017 that runs on our Coco dataset. Ignore fields that are note useful
    :input_data_dir: Where the original data is stored
    :dest_data_dir: Where the resulting data should be stored
    """
    input_data_dir = Path(input_data_dir)
    dest_data_dir = Path(dest_data_dir)
    _copy_to_new_dir("train", 1000, input_data_dir, dest_data_dir)
    _copy_to_new_dir("val", 500, input_data_dir, dest_data_dir)


def _copy_to_new_dir(mode: str, n_images: int, input_data_dir: Path, dest_data_dir: Path):
    """Copy either train or val from input dir into destination dir
    :param mode: Either "train" or "val"
    :param n_images: How many images/annotations to copy for this mode
    :input_data_dir: Where the original data is stored
    :dest_data_dir: Where the resulting data should be stored
    """
    input_instances_path = input_data_dir / "annotations" / f"instances_{mode}2017.json"

    dest_annotation_folder = dest_data_dir / "annotations"
    dest_annotation_folder.mkdir(exist_ok=True, parents=True)
    dest_instances_path = dest_annotation_folder / f"instances_{mode}2017.json"

    with open(input_instances_path, "r") as f:
        instances = json.load(f)

    image_ids = {instance["id"] for instance in instances["images"]}
    annotation_image_ids = {instance["image_id"] for instance in instances["annotations"]}

    kept_image_ids = list(image_ids & annotation_image_ids)[:n_images]  # Make sure that the ids taken include both image and annotation
    kept_annotations = [image for image in instances["annotations"] if image["image_id"] in kept_image_ids]
    kept_images = [image for image in instances["images"] if image["id"] in kept_image_ids]

    instances["images"] = kept_images
    instances["annotations"] = kept_annotations
    kept_images_name = [image["file_name"] for image in instances["images"]]

    input_images_dir = input_data_dir / "images" / f"{mode}2017"
    dest_images_dir = dest_data_dir / "images" / f"{mode}2017"
    dest_images_dir.mkdir(exist_ok=True, parents=True)

    with open(dest_instances_path, "w") as f:
        json.dump(instances, f)

    for image_name in kept_images_name:
        shutil.copy(str(input_images_dir / image_name), str(dest_images_dir / image_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a sub set of Coco into specified dir")
    parser.add_argument("--input_data_dir", help="Where the full coco dataset is stored", default="/data/coco")
    parser.add_argument("--dest_data_dir", help="Where the resulting data should be stored", required=True)
    args = parser.parse_args()
    copy_train_val_to_new_dir(args.input_data_dir, args.dest_data_dir)
