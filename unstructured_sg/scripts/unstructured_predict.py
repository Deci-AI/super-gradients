"""
Script to run prediction on directory containing images or PDF files using a model wrapped in a super-gradient pipeline.

Example usage:
PYTHONPATH=. HF_TOKEN=<token> python unstructured_sg/scripts/unstructured_predict.py \
--model_name yolox_mar24_2_1 --input_dir /path/to/input_dir --output_dir /path/to/output_dir

Run `python unstructured_sg/scripts/unstructured_predict.py --help` to see all options and supported model names.

HF_TOKEN environment variable must be set in order to download the model from the Hugging Face Hub. 
"""
import json
import os
import logging

import dotenv

import click
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm

from unstructured_sg.model_configs import MODEL_CONFIGS
from unstructured_sg.utils import dump_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_name):
    """
    Load the model specified by model_name.
    """
    from super_gradients.training.models import get  # Local import to delay heavy imports

    dotenv.load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    model_config = MODEL_CONFIGS[model_name]
    checkpoint_path = hf_hub_download(repo_id=model_config.checkpoint_repo_id,
                                      filename=model_config.checkpoint_filename)

    model = get(model_name=model_config.model_name, num_classes=model_config.num_classes,
                checkpoint_path=checkpoint_path)

    return model


def process_image_files(model, files, input_dir, output_dir, output_json):
    """Process each file in the provided list of files."""
    tqdm_pbar = tqdm(files, desc="Processing files")
    for file_name in tqdm_pbar:
        tqdm_pbar.set_description(f"Processing {file_name}")
        process_single_file(model, file_name, input_dir, output_dir, output_json=output_json)


def process_single_file(model, file_name, input_dir, output_dir, output_json):
    """Process a single image or PDF file."""
    file_path = input_dir / file_name

    images = load_images(file_path)
    if not images:
        logging.warning(f"No images found or could not load: {file_path}")
        return

    save_predictions(model, images, file_name, output_dir, output_json)


def load_images(file_path):
    """Load images from a file path, converting PDFs as needed."""
    if file_path.suffix.lower() == ".pdf":
        images = convert_from_path(file_path)
        return [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image in images]

    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        return []

    return [image]


def save_predictions(model, images, file_name, output_dir, output_json):
    """Run model prediction on images and save the outputs."""
    for page_count, image_array in enumerate(images):
        output = model.predict(image_array)
        image_output = output.draw()

        suffix = Path(file_name).suffix
        output_image_path = output_dir / f"{file_name.removesuffix(suffix)}_{page_count}.png"
        cv2.imwrite(str(output_image_path), image_output)
        logging.debug(f"Saved in: {output_image_path}.")

        if output_json:
            save_json_output(output, output_image_path)


def get_dict_output(output):
    """Get output prediction details in dictionary format."""
    dict_output = {
        "class_names": output.class_names,
        "bboxes_xyxy": output.prediction.bboxes_xyxy.tolist(),
        "confidence": output.prediction.confidence.tolist(),
        "labels": output.prediction.labels.tolist(),
        "image_shape": output.prediction.image_shape,
    }
    return dict_output


def save_json_output(output, output_path):
    """Save output prediction details in JSON format."""
    out_json_path = output_path.with_suffix(".json")
    dict_output = get_dict_output(output)
    dump_json(path=out_json_path, data=dict_output)
    logging.debug(f"Saved in: {out_json_path}")


@click.command()
@click.option("--model_name", default="yolox_mar24_2_1", type=click.Choice(MODEL_CONFIGS.keys()),
              help="Model name to use. Note that it's registered pipeline name, not some architecture name.")
@click.option("--input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
              required=True, help="Path to directory with input images / documents.")
@click.option("--output_dir", type=click.Path(file_okay=False, dir_okay=True, resolve_path=True), required=True,
              help="Path to directory where results should be saved.")
@click.option("--split_info_pth", default=None, type=click.Path(exists=True, dir_okay=False, resolve_path=True),
              help="Optional path to COCO output json annotation file. Not necessary for casual use.")
@click.option("--output_json", is_flag=True, help="If True, saves results in json format.")
def main(model_name, input_dir, output_dir, split_info_pth, output_json):
    logging.info("Starting processing...")
    logging.info(f"Model: {model_name}, input_dir: {input_dir}, output_dir: {output_dir}")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_name)

    if split_info_pth:
        with open(split_info_pth, "r") as file:
            split_info = json.load(file)
        files = [image["file_name"] for image in split_info["images"]]
    else:
        files = [file.name for file in input_dir.iterdir()]

    process_image_files(model, files, input_dir, output_dir, output_json)
    logging.info("Processing completed.")


if __name__ == "__main__":
    main()
