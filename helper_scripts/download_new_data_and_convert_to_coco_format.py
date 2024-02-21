"""
This is a helper scripts that downloads new dataset images based on annotation file,
and converts annotations as .json file in COCO format: https://cocodataset.org/#format-data.

>> python helper_scripts/download_new_data_and_convert_to_coco_format.py --images_dir /path/to/images/dir \
    --labels_path /path/to/original/labels.txt  --coco_labels_dir /path/to/coco/labels.json
"""
import argparse
import os
import random
import urllib.request
from math import ceil
from pathlib import Path

from convert_mini_holistic_to_coco_format import check_and_add_category, check_and_add_image, get_id_from_dict_list
from PIL import Image
from tqdm import tqdm
from utils import dump_json, load_json_by_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert dataset to standard COCO format.")

    parser.add_argument(
        "--images_dir",
        type=Path,
        help="Path to directory where documents are stored.",
    )
    parser.add_argument(
        "--labels_path",
        type=Path,
        help="Path to Label Studio json annotation file.",
    )
    parser.add_argument(
        "--coco_labels_dir",
        type=Path,
        help="Path to directory where COCO annotations will be saved.",
    )
    return parser.parse_args()


def download_images(labels: list[dict], images_dir: Path):
    for label in tqdm(labels, desc="Downloading images..."):
        if not os.path.exists(f"{images_dir}/{label['url'].split('/')[-1]}"):
            urllib.request.urlretrieve(label["url"], f"{images_dir}/{label['url'].split('/')[-1]}")


def flatten_tuple(input_tuple):
    flattened_list = []

    def flatten_recursive(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                flatten_recursive(item)
        else:
            flattened_list.append(value)

    flatten_recursive(input_tuple)
    return tuple(flattened_list)


def filter_dicts_by_keys(input_list, keys_to_check):
    seen_values = set()
    result = []

    for item in input_list:
        values_to_check = tuple(item[key] for key in keys_to_check)
        values_to_check = flatten_tuple(values_to_check)

        if values_to_check not in seen_values:
            result.append(item)
            seen_values.add(values_to_check)

    return result


def main(
    images_dir: Path,
    labels_path: Path,
    coco_labels_dir: Path,
    seed: int = 42,
) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    documents = load_json_by_line(labels_path)

    random.seed(seed)
    random.shuffle(documents)

    download_images(documents, images_dir)

    COCO_anno = {
        "images": [],
        "categories": [],
        "annotations": [],
        "info": {
            "year": 2024,
            "version": "1.0",
        },
    }

    COCO_image_id = 0
    COCO_category_id = 0
    COCO_annotation_id = 0

    for document in tqdm(documents, desc="Preparing COCO annotations..."):
        document_file_name = document["url"].split("/")[-1]
        document_file_pth = f"{images_dir}/{document_file_name}"

        if os.path.exists(document_file_pth):
            try:
                target_image = Image.open(document_file_pth)
            except Exception as e:
                print(f"Can't open image {document_file_pth}: {e}")
                continue

            COCO_anno, COCO_image_id = check_and_add_image(document_file_name, target_image, COCO_image_id, document["sd_result"]["id"], COCO_anno)

            for item in document["sd_result"]["items"]:
                bbox = item["meta"]["geometry"]
                label = item["labels"]["entity"].lower()

                COCO_anno, COCO_category_id = check_and_add_category(label, COCO_anno, COCO_category_id)

                image_id = get_id_from_dict_list(COCO_anno["images"], "file_name", document_file_name)
                category_id = get_id_from_dict_list(COCO_anno["categories"], "name", label)

                if item["meta"]["type"] == "POLYGON":
                    X = []
                    Y = []
                    for pair in bbox:
                        X.append(pair[0])
                        Y.append(pair[1])
                    x0 = min(X)
                    x1 = max(X)
                    y0 = min(Y)
                    y1 = max(Y)
                    width = x1 - x0
                    height = y1 - y0
                elif item["meta"]["type"] == "BBOX":
                    x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
                    width = x1 - x0
                    height = y1 - y0
                area = width * height

                COCO_anno["annotations"].append(
                    {
                        "id": COCO_annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": [],
                        "bbox": [float(x0), float(y0), float(width), float(height)],
                        "ignore": 0,
                        "iscrowd": 0,
                        "area": float(area),
                    },
                )
                COCO_annotation_id += 1

    keys_to_check = ["image_id", "category_id", "bbox"]
    COCO_anno["annotations"] = filter_dicts_by_keys(COCO_anno["annotations"], keys_to_check)

    coco_labels_path = coco_labels_dir / "all.json"
    dump_json(coco_labels_path, COCO_anno)
    print(f"Annotation saved in {coco_labels_path}")

    number_of_documents = len(COCO_anno["images"])
    print(f"There are {number_of_documents} documents found.")

    print("Preparing train/val/test splits...")
    splits = {"train": 0.7, "val": 0.2, "test": 0.1}
    start = 0
    for split in splits:
        number_of_samples = ceil(splits[split] * number_of_documents)
        COCO_anno_split = {}
        COCO_anno_split["categories"] = COCO_anno["categories"]

        stop = start + number_of_samples
        if stop > number_of_documents:
            stop = number_of_documents
        COCO_anno_split["images"] = COCO_anno["images"][start:stop]
        start = stop
        print(f"{len(COCO_anno_split['images'])} samples in {split} split.")

        img_ids = [image["id"] for image in COCO_anno_split["images"]]
        COCO_anno_split["annotations"] = []
        for anno in tqdm(COCO_anno["annotations"]):
            if anno["image_id"] in img_ids:
                COCO_anno_split["annotations"].append(anno)

        coco_labels_path = coco_labels_dir / f"{split}.json"
        dump_json(coco_labels_path, COCO_anno_split)
        print(f"{split} finished. Annotation saved in {coco_labels_path}")

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.images_dir,
        args.labels_path,
        args.coco_labels_dir,
    )
