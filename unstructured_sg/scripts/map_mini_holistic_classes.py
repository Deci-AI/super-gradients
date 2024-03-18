from pathlib import Path

from unstructured_sg.mappings import MINIHOLISTIC_CAT, UNSTRUCTURED_CAT, MINIHOLISTIC_UNSTRUCTURED_MAP
from unstructured_sg.utils import load_json


COCO_PATH = Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic/COCO/test.json")
COCO_CONVERTED_PATH = Path("/mnt/ml-team/homes/marianna.parzych/Unstructured/MiniHolistic/COCO/test_unstructured_classes.json")

COCO_anno = load_json(COCO_PATH)

table_count = 0
form_count = 0
image_count = 0
formula_count = 0

COCO_anno["categories"] = UNSTRUCTURED_CAT
mapped_annotations = []
for annotation in COCO_anno["annotations"]:
    original_category = [cat for cat in MINIHOLISTIC_CAT if cat["id"] == annotation["category_id"]][0]
    mapped_cat_name = MINIHOLISTIC_UNSTRUCTURED_MAP[original_category["name"]]
    mapped_category = [cat for cat in UNSTRUCTURED_CAT if cat["name"] == mapped_cat_name][0] if mapped_cat_name is not None else None
    mapped_id = mapped_category["id"] if mapped_category is not None else None
    if mapped_id is not None:
        annotation["category_id"] = mapped_id
        mapped_annotations.append(annotation)
    if mapped_cat_name == 'table':
        table_count += 1
    elif mapped_cat_name == 'form':
        form_count += 1
    elif mapped_cat_name == 'image':
        image_count += 1
    elif mapped_cat_name == 'formulas':
        formula_count += 1
COCO_anno["annotations"] = mapped_annotations

#dump_json(COCO_CONVERTED_PATH, COCO_anno)

print("Done")
