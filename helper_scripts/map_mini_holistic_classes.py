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
    {"supercategory": "Text", "id": 10, "name": "Text"},
    {"supercategory": "Title", "id": 11, "name": "Title"},
]

UNSTRUCTURED_CAT = [
    {'id': 0, 'name': 'image'},
    {'id': 1, 'name': 'page_number'},
    {'id': 2, 'name': 'paraprgaphs_in_image'},
    {'id': 3, 'name': 'paraprgaph'},
    {'id': 4, 'name': 'subheading'},
    {'id': 5, 'name': 'page_header'},
    {'id': 6, 'name': 'formulas'},
    {'id': 7, 'name': 'other'},
    {'id': 8, 'name': 'table'},
    {'id': 9, 'name': 'page_footer'},
    {'id': 10, 'name': 'title'},
    {'id': 11, 'name': 'form'},
    {'id': 12, 'name': 'paraprgaphs_in_form'},
    {'id': 13, 'name': 'checkbox_checked'},
    {'id': 14, 'name': 'checkbox'},
    {'id': 15, 'name': 'radio_button'},
    {'id': 16, 'name': 'radio_button_checked'},
]

MINIHOLISTIC_CAT = [
    {'id': 0, 'name': 'Text'},
    {'id': 1, 'name': 'Table'},
    {'id': 2, 'name': 'NarrativeText'},
    {'id': 3, 'name': 'Title'},
    {'id': 4, 'name': 'Footer'},
    {'id': 5, 'name': 'PageNumber'},
    {'id': 6, 'name': 'Header'},
    {'id': 7, 'name': 'ListItem'},
    {'id': 8, 'name': 'Figure'},
    {'id': 9, 'name': 'OtherText'},
    {'id': 10, 'name': 'MajorTitle'},
    {'id': 11, 'name': 'FieldName'},
    {'id': 12, 'name': 'Value'},
    {'id': 13, 'name': 'Form'},
    {'id': 14, 'name': 'Image'},
    {'id': 15, 'name': 'Caption'},
    {'id': 16, 'name': 'List'},
    {'id': 17, 'name': 'Checkbox'},
    {'id': 18, 'name': 'Abstract'},
    {'id': 19, 'name': 'Formula'},
    {'id': 20, 'name': 'PageBreak'},
]

MINIHOLISTIC_DOCLAYNET_MAP = {
    "Text": None,
    "Table": None,
    "NarrativeText": None,
    "Title": None,
    "Footer": None,
    "PageNumber": None,
    "Header": None,
    "ListItem": None,
    "Figure": None,
    "OtherText": None,
    "MajorTitle": None,
    "FieldName": None,
    "Value": None,
    "Form": None,
    "Image": None,
    "Caption": None,
    "List": None,
    "Checkbox": None,
    "Abstract": None,
    "Formula": None,
    "PageBreak": None,
}

MINIHOLISTIC_UNSTRUCTURED_MAP = {
    "Text": None,
    "Table": 'table',
    "NarrativeText": None,
    "Title": None,
    "Footer": None,
    "PageNumber": None,
    "Header": None,
    "ListItem": None,
    "Figure": None,
    "OtherText": None,
    "MajorTitle": None,
    "FieldName": None,
    "Value": None,
    "Form": 'form',
    "Image": 'image',
    "Caption": None,
    "List": None,
    "Checkbox": None,
    "Abstract": None,
    "Formula": 'formulas',
    "PageBreak": None,
}

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
