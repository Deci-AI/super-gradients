import argparse
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from utils import load_json, dump_json


# Known issue with labels:
# OptimalEstimationMethodologies-for-PanelDataRegressionModels-pg9-12.pdf has 4 pages,
# LS annotations point to pages 250-253

FP_MAPPING = {
    "wisconsin-sample-license.pdf": "wisconsin-sample-license.jpeg",
    "scannedpaper1-jpeg.pdf": "scannedpaper1-jpeg.jpg",
    "scanned letter.pdf": "scanned_letter.png",
    "receipt-B.pdf": "receipt-B.jpg",
    "receipt-A.pdf": "receipt-A.jpg",
    "LCD data table.pdf": "LCD_data_table.jpg",
    "income-18445487 1.pdf": "income-18445487_1.jpg",
    "example table.pdf": "example_table.jpg",
    "cashflow-18445494 2.pdf": "cashflow-18445494_2.jpg",
    "balance-18460658 57.pdf": "balance-18460658_57.jpg",
    # "NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-Volume3-MassPr.pdf":  # ambigous, there are 2 similar files:
    # NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-Volume3-MassProperties-Pg54.pdf
    # NASA-SNA-8-D-027III-Rev2-CsmLmSpacecraftOperationalDataBook-Volume3-MassProperties-pg856.pdf
    "intel-extension-for-transformers intel extension for transformers llm runt.pdf": "intel-extension-for-transformers intel_extension_for_transformers llm runtime graph docs infinite_in.pdf",
    "intel intel-extension-for-transformers  Build your chatbot within minutes .pdf": "intel_intel-extension-for-transformers_ Build your chatbot within minutes on your favorite device.pdf",
    "Huang Improving Table Structure Recognition With Visual-Alignment Sequenti.pdf": "Huang_Improving_Table_Structure_Recognition_With_Visual-Alignment_Sequential_Coordinate_Modeling_CVPR_2023_paper-p6.pdf",
    "scanned letter.pdf": "scanned_letter.png",
    "LCD data table.pdf": "LCD_data_table.jpg",
    "income-18445487 1.pdf": "income-18445487_1.jpg",
    "example table.pdf": "example_table.jpg",
    "cashflow-18445494 2.pdf": "cashflow-18445494_2.jpg",
    "balance-18460658 57.pdf": "balance-18460658_57.jpg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert mini-holistic dataset with Label Studio annotations to standard COCO format.")

    parser.add_argument(
        "--docs_dir",
        default=Path("/mnt/ml-team/unstructured/DATA/dvc-data-registry/holistic-mini-pdf-image-dataset/mini-holistic-all/src"),  # todo remove
        type=Path,
        help="Path to directory where documents are stored.",
    )
    parser.add_argument(
        "--ls_labels_path",
        default=Path(
            "/mnt/ml-team/unstructured/DATA/dvc-data-registry/holistic-mini-pdf-image-dataset/mini-holistic-all/ls/export_45956_project-45956-at-2024-01-10-23-16-24cfbda6.json"
        ),  # todo remove
        type=Path,
        help="Path to Label Studio json annotation file.",
    )
    parser.add_argument(
        "--output_img_dir",
        default=Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic/PNG"),  # todo remove
        type=Path,
        help="Path to directory where documents are stored.",
    )
    parser.add_argument(
        "--coco_labels_path",
        default=Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic/COCO/test.json"),  # todo remove
        type=Path,
        help="Path to COCO output json annotation file.",
    )
    return parser.parse_args()


def get_id_from_dict_list(dict_list, key, value):
    for dict_ in dict_list:
        if dict_[key] == value:
            return dict_["id"]


def main(
    docs_dir: Path,
    ls_labels_path: Path,
    output_img_dir: Path,
    coco_labels_path: Path,
) -> None:
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

    all_annotations = load_json(ls_labels_path)

    for file_annotation in all_annotations:
        file_upload = file_annotation["file_upload"]
        file_name = "-".join(file_upload.split("-")[1:])
        extensions = [".pdf", ".jpg", ".jpeg", ".png"]
        extension = [extension for extension in extensions if extension in file_name]
        if extension:
            extension = extension[0]
        else:
            print(f"Not known extension of the file name: {file_name}")
        file_stem, suffix = file_name.split(extension)
        if suffix:
            page_id = int(suffix.split(".jpg")[0].split("_")[-1])
        else:
            page_id = 0
        file_path = docs_dir / f"{file_stem}.pdf"
        # todo: more mappings and cleanup
        if not file_path.exists():
            mapped = FP_MAPPING.get(file_path.name, None)
            if mapped:
                file_path = Path(FP_MAPPING.get(file_path.name, None))
            else:
                file_path = Path(str(file_path).replace("_", " "))
                if not file_path.exists():
                    print(f"{file_path} doesn't exist.")
                    continue
        if file_path.suffix == ".pdf":
            pdf_images = convert_from_path(file_path)
            if len(pdf_images) < 1:
                print(f"Found 0 images in file: {file_path}")
                continue
            if page_id > len(pdf_images) - 1:
                print(f"Page id {page_id} is too high for file: {file_path} with {len(pdf_images)} pages.")
                continue
            output_img = pdf_images[page_id]
            output_img_name = f"{file_path.name.replace('.pdf','')}_{page_id}.jpg"
        else:
            page_id = 0
            file_path = docs_dir / file_path
            output_img = Image.open(file_path)
            output_img_name = file_path.name

        if output_img_name not in [image["file_name"] for image in COCO_anno["images"]]:
            width, height = output_img.size
            COCO_anno["images"].append({"width": width, "height": height, "id": COCO_image_id, "ls_id": file_annotation["id"], "file_name": output_img_name})
            COCO_image_id += 1

        output_image_path = output_img_dir / output_img_name
        output_img.save(output_image_path)
        print(f"Saved img: {output_image_path}")

        for element in file_annotation["annotations"]:
            for annotation in element["result"]:
                if annotation["type"] == "labels":
                    category = annotation["value"]["labels"][0]
                    if category not in [cat["name"] for cat in COCO_anno["categories"]]:
                        COCO_anno["categories"].append({"id": COCO_category_id, "name": category})
                        COCO_category_id += 1

                    x0 = annotation["value"]["x"] / 100 * annotation["original_width"]
                    y0 = annotation["value"]["y"] / 100 * annotation["original_height"]
                    width = annotation["value"]["width"] / 100 * annotation["original_width"]
                    height = annotation["value"]["height"] / 100 * annotation["original_height"]
                    area = width * height

                    image_id = get_id_from_dict_list(COCO_anno["images"], "file_name", output_img_name)
                    category_id = get_id_from_dict_list(COCO_anno["categories"], "name", category)

                    COCO_anno["annotations"].append(
                        {
                            "id": COCO_annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [],
                            "bbox": [x0, y0, width, height],
                            "ignore": 0,
                            "iscrowd": 0,
                            "area": area,
                        },
                    )
                    COCO_annotation_id += 1

    dump_json(coco_labels_path, COCO_anno)
    print(f"Annotation saved in {coco_labels_path}")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.docs_dir,
        args.ls_labels_path,
        args.output_img_dir,
        args.coco_labels_path,
    )
