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
from utils import dump_json, load_json, load_txt_with_json, load_txt, load_csv_with_json


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
        nargs="+",
        help="Paths to Label Studio json annotation file.",
    )
    parser.add_argument(
        "--existing_coco_labels_dir",
        type=Path,
        help="Path to directory where existing COCO splits are saved.",
    )
    parser.add_argument(
        "--existing_categories_file",
        type=Path,
        default=None,
        help="Path to file where list of categories is stored. If None, categories will be generated based on labels.",
    )
    parser.add_argument(
        "--train_split_name",
        default="train.json",
        type=str,
        help="Path to directory where existing COCO splits are saved.",
    )
    parser.add_argument(
        "--val_split_name",
        default="val.json",
        type=str,
        help="Path to directory where existing COCO splits are saved.",
    )
    parser.add_argument(
        "--test_split_name",
        default="test.json",
        type=str,
        help="Path to directory where existing COCO splits are saved.",
    )
    parser.add_argument(
        "--coco_labels_dir",
        type=Path,
        help="Path to directory where COCO annotations will be saved.",
    )
    parser.add_argument(
        "--train_split_size",
        default=0.7,
        type=float,
        help="What fraction of documents should be included in the training split.",
    )
    parser.add_argument(
        "--val_split_size",
        default=0.2,
        type=float,
        help="What fraction of documents should be included in the validatio split.",
    )
    parser.add_argument(
        "--test_split_size",
        default=0.1,
        type=float,
        help="What fraction of documents should be included in the testing split.",
    )
    parser.add_argument("--download_images", action="store_true", help="Flag indicating images should be downloaded from S3 bucket.")
    parser.add_argument("--use_existing_COCO", action="store_true", help="Flag indicating the previously converted annotations should be used to split.")
    return parser.parse_args()


def sort_labels(labels: list[dict]) -> list[dict]:
    for indice in range(len(labels)):
        labels[indice]["file_name"] = labels[indice]["url"].split("/")[-1]
    labels = sorted(labels, key=lambda x: x["file_name"])
    return labels


def download_images_from_url(labels: list[dict], images_dir: Path):
    for label in tqdm(labels, desc="Downloading images..."):
        output_path = f"{images_dir}/{label['url'].split('/')[-1]}"
        if not Path(output_path).exists():
            try:
                # Fix some unsupported chars in url
                label_url = label["url"].replace(" ", "+").replace("®", "%C2%AE")
                urllib.request.urlretrieve(label_url, output_path)
            except Exception as e:
                # Some of the files from FinanceBench are not found due to new location
                updated_url = label_url.replace(
                    "http://utic-dev-tech-fixtures.s3.us-east-2.amazonaws.com/data_for_external_3rd_party_chinese_company/FinanceBench/",
                    "http://utic-dev-tech-fixtures.s3.us-east-2.amazonaws.com/data_for_external_3rd_party_chinese_company/SourceImages/FinanceBench/",
                )
                try:
                    urllib.request.urlretrieve(updated_url, output_path)
                except:  # noqa: E722
                    print(f"Coudn't download image: {label['url']}.\n{e}")


def load_images_names_for_split(split_path: Path) -> list[str] | None:
    if split_path.exists():
        split = load_json(split_path)
        return [image["file_name"] for image in split["images"]]
    else:
        return None


def load_existing_splits(
    existing_coco_labels_dir: Path,
    train_split_name: str,
    val_split_name: str,
    test_split_name: str,
) -> dict:
    if existing_coco_labels_dir:
        train_split_path = existing_coco_labels_dir / train_split_name
        val_split_path = existing_coco_labels_dir / val_split_name
        test_split_path = existing_coco_labels_dir / test_split_name

        splits = {
            "train": load_images_names_for_split(train_split_path),
            "val": load_images_names_for_split(val_split_path),
            "test": load_images_names_for_split(test_split_path),
        }
        return splits
    else:
        return {
            "train": None,
            "val": None,
            "test": None,
        }


def init_categories(existing_categories_file: Path, COCO_anno: dict, COCO_category_id: int):
    categories_str = load_txt(existing_categories_file)
    categories_list = categories_str[0].replace(",", "").split(" ")
    categories_list = [element for element in categories_list if element != ""]
    COCO_categories = []
    for cat in categories_list:
        COCO_categories.append({"id": COCO_category_id, "name": cat})
        COCO_category_id += 1
    COCO_anno["categories"] = COCO_categories
    return COCO_anno, COCO_category_id


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


def filter_dicts_by_keys(input_dicts, keys_to_check):
    seen_values = set()
    result = []

    for item in input_dicts:
        values_to_check = tuple(item[key] for key in keys_to_check)
        values_to_check = flatten_tuple(values_to_check)

        if values_to_check not in seen_values:
            result.append(item)
            seen_values.add(values_to_check)

    return result


def clean_categories(categories: list[dict]) -> list[dict]:
    # Replace "paraprgaph" with "paragraph"
    for category in categories:
        category["name"] = category["name"].replace("paraprgaph", "paragraph")
    return sorted(categories, key=lambda x: x["id"])


def save_list_to_txt_file(path: Path, data: list[str]):
    with open(path, "w") as file:
        for name in data:
            file.write(name + ", ")


def main(
    images_dir: Path,
    labels_paths: list[Path],
    existing_coco_labels_dir: Path,
    existing_categories_file: Path,
    train_split_name: str,
    val_split_name: str,
    test_split_name: str,
    coco_labels_dir: Path,
    train_split_size: float,
    val_split_size: float,
    test_split_size: float,
    download_images: bool,
    use_existing_COCO: bool,
    seed: int = 42,
) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    coco_labels_dir.mkdir(parents=True, exist_ok=True)

    documents = []
    for path in labels_paths:
        if path.suffix == ".csv":
            documents.extend(load_csv_with_json(path))
        elif path.suffix == ".txt":
            documents.extend(load_txt_with_json(path))

    # Download missing images
    if download_images:
        download_images_from_url(documents, images_dir)

    # Ensure reproducibility between different label files
    documents = sort_labels(documents)
    random.seed(seed)
    random.shuffle(documents)

    coco_labels_path = coco_labels_dir / "all.json"
    if use_existing_COCO and coco_labels_path.exists():
        COCO_anno = load_json(coco_labels_path)
    else:
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

        if existing_categories_file and len(COCO_anno["categories"]) == 0:
            COCO_anno, COCO_category_id = init_categories(existing_categories_file, COCO_anno, COCO_category_id)

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
        COCO_anno["categories"] = clean_categories(COCO_anno["categories"])

        # Create a list of class names to use in training configs
        class_names = [category["name"] for category in COCO_anno["categories"]]
        class_names_path = coco_labels_dir / "classnames.txt"
        save_list_to_txt_file(class_names_path, class_names)

        dump_json(coco_labels_path, COCO_anno)
        print(f"Annotation saved in {coco_labels_path}")

    print("Preparing train/val/test splits...")
    splits = {"train": train_split_size, "val": val_split_size, "test": test_split_size}
    existing_splits = load_existing_splits(existing_coco_labels_dir, train_split_name, val_split_name, test_split_name)
    all_COCO_anno_splits = {}

    # If existing split provided - use them
    for split in splits:
        COCO_anno_split = {}
        COCO_anno_split["categories"] = COCO_anno["categories"]
        COCO_anno_split["images"] = []

        if existing_splits[split]:
            for doc_name in existing_splits[split]:
                entry = next((image for image in COCO_anno["images"] if image["file_name"] == doc_name), None)

                if entry:
                    COCO_anno_split["images"].append(entry)
                    COCO_anno["images"].remove(entry)
        all_COCO_anno_splits[split] = COCO_anno_split

    # Add new images to the split and complete the annotations
    start = 0
    number_of_remaining_documents = len(COCO_anno["images"])
    for split in splits:
        COCO_anno_split = all_COCO_anno_splits[split]
        if number_of_remaining_documents:
            number_of_samples = ceil(splits[split] * number_of_remaining_documents)

            # if number_of_samples > len(COCO_anno_split["images"]):
            stop = start + number_of_samples
            if stop > number_of_remaining_documents:
                stop = number_of_remaining_documents
            COCO_anno_split["images"].extend(COCO_anno["images"][start:stop])
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
        args.existing_coco_labels_dir,
        args.existing_categories_file,
        args.train_split_name,
        args.val_split_name,
        args.test_split_name,
        args.coco_labels_dir,
        args.train_split_size,
        args.val_split_size,
        args.test_split_size,
        args.download_images,
        args.use_existing_COCO,
    )
