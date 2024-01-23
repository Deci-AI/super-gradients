from pathlib import Path

from utils import load_json, dump_json

DOCLAYNET_CAT = [
    {"supercategory": "Caption", "id": 1, "name": "Caption"},
    {"supercategory": "Footnote", "id": 2, "name": "Footnote"},
    {"supercategory": "Formula", "id": 3, "name": "Formula"},
    {"supercategory": "List-item", "id": 4, "name": "List-item"},
    {"supercategory": "Page-footer", "id": 5, "name": "Page-footer"},
    {"supercategory": "Page-header", "id": 6, "name": "Page-header"},
    {"supercategory": "Picture", "id": 7, "name": "Picture"},
    {"supercategory": "Section-header", "id": 8, "name": "Section-header"},
    {"supercategory": "Table", "id": 9, "name": "Table"},
]

MINIHOLISTIC_DOCLAYNET_MAP = {
    0: 10,
    1: 9,
    2: 10,
    3: 11,
    4: 5,
    5: 5,
    6: 6,
    7: 4,
    8: 7,
    9: 10,
    10: 11,
    11: None,
    12: None,
    13: None,
    14: 7,
    15: 1,
    16: 4,
    17: None,
    18: 10,
    19: 3,
    20: None,
    # None: 2, 'name': 'Footnote'
    # None: 8, 'name': 'Section-header'
}

COCO_PATH = Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic/COCO/test.json")
COCO_CONVERTED_PATH = Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic/COCO/test_doclaynet_classes.json")

COCO_anno = load_json(COCO_PATH)

COCO_anno["categories"] = DOCLAYNET_CAT
mapped_annotations = []
for annotation in COCO_anno["annotations"]:
    mapped_id = MINIHOLISTIC_DOCLAYNET_MAP[annotation["category_id"]]
    if mapped_id is not None:
        annotation["category_id"] = mapped_id
        mapped_annotations.append(annotation)
COCO_anno["annotations"] = mapped_annotations

dump_json(COCO_CONVERTED_PATH, COCO_anno)

print("Done")
