# README - unstructured.io in super-gradients

## Setup
1. Clone the repository
2. Install dependencies
    1. `pip install -e .`
3. Create a copy of _.env.template_ file and name it _.env_. Fill it with proper values.
4. To get more information on Neptune.ai / modal usage, please refer to the respective sections below.


## Running inference (simple usage for users out of DS team)
To run inference using trained model, you can use `unstructured_sg/scripts/unstructured_predict.py` script.
Example usage:
```
PYTHONPATH=. HF_TOKEN=<token> python unstructured_sg/scripts/unstructured_predict.py --input_dir <path_to_image> --output_dir <path_to_output>
```
Run `PYTHONPATH=. python unstructured_sg/scripts/unstructured_predict.py --help` to see all available options and if you'd like to change the model version.


## Typical workflow
Below there's an example of a workflow for working with unstructured data in super-gradients. The example is based on training and evaluating YOLOX model on unstructured data.
Please note that if you need some customizations, you may need to adjust the steps accordingly.

1. Download data and prepare it for training
    1. `python unstructured_sg/scripts/download_new_data_and_convert_to_coco_format.py --help`
2. Optionally, if using modal, upload data to modal volume
    1. `modal volume put od_datasets <local_path> <remote_path>`
3. Update configs if necessary:
    1. dataset parameters: `src/super_gradients/recipes/dataset_params/unstructured_feb24_modal_dataset_params.yaml`
    2. experiment parameters: `src/super_gradients/recipes/train_unstructured_feb24_yolox.yaml`
4. Run training
    1. `modal run src/super_gradients/train_from_recipe_modal.py --config-name train_unstructured_feb24_yolox` OR locally
    2. `python src/super_gradients/train_from_recipe.py --config-name train_unstructured_feb24_yolox`
    3. Training logs are available in neptune.ai (if enabled) and in modal (if it's used).
5. Optionally, if using modal, download checkpoints from modal volume. Checkpoints can be also downloaded manually from neptune.ai.
    1. `modal volume get checkpoints <remote_path> <local_path>`
6. Run evaluation
    1. Update experiment parameters: `src/super_gradients/recipes/train_unstructured_feb24_yolox.yaml` (set checkpoint path to downloaded checkpoint)
    2. Update dataset parameters if you'd like to change split or use different data
    3. `python src/super_gradients/evaluate_from_recipe.py --config-name train_unstructured_feb24_yolox`
    4. Evaluation logs are available in neptune.ai (if enabled).
7. If you want to visualize results, you can use `unstructured_sg/scripts/visualize_and_save_results_from_directory.py` script to visualize results using checkpoint.
8. To optimize thresholds for given model, you can use `src/super_gradients/scripts/find_detection_score_threshold.py` script.
9. If you want to push the model for further release or tests follow these steps:
    1. Update `src/super_gradients/training/models/detection_models/yolo_base.py` to include your model, use `class YoloX_MAR24_1_1` as an example.
    2. Add model into `src/super_gradients/training/models/detection_models/yolox.py`, use `YoloX_L_MAR24_1_1` as an example.
    3. Add model into `src/super_gradients/common/object_names.py`, use `YOLOX_MAR24_1_1` as an example.
    4. Upload checkpoint to huggingface unstructured-io organization.
    5. Add model config into `unstructured_sg/model_configs.py`
10. Now you can run inference using `unstructured_sg/scripts/unstructured_predict.py --model_name <new_model_name> ...` script.


## Neptune integration
The repository provides basic integration with neptune.ai. To track your experiment in neptune.ai yopu have to set `neptune_logging` parameter in training hyperparams. 
Example:
```
training_hyperparams:
  neptune_logging:
    name:"lotus-alligator"
    tags:["quickstart", "script"]
```
Project name and API token are set based on environment variables. Prepare your _.env_ based on _.env.template_ (this sets Set `NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`).
`NEPTUNE_API_TOKEN` may be found in your user profile setting in neptune.ai.

## Modal

### Setup
After you are added into _unstructured-io_ workspace on modal - run `modal setup` command to authorize and create token automatically.

### Environment variables
Prior to running experiments, prepare your _.env_ based on _.env.template_ file (copy _.env.template_ to _.env_ and set variables accordingly). 
_train_from_recipe_modal.py_ creates modal secrets from dotenv files, so setting environment variables in other way will not work properly.

### Run experiment
To run training using modal you can run _train_from_recipe_modal.py_ script:
`modal run src/super_gradients/train_from_recipe_modal.py --config-name train_unstructured_jan24_yolox`

Carefully review _train_from_recipe_modal.py_ script when you want to make some changes - some things are hardcoded there (like branch to be used, data volume or checkpoints volume) 

### Volumes
To handle modal volumes, CLI `modal volume` is useful. Run `modal volume --help` to see what operations are supported. You can easily upload data (so that it's available for training), download checkpoints, list or remove volumes.

For now in _unstructured-io_ space we actively use following volumes:

* **od_datasets** - with all the dataset versions gathered by unstructured in 2024
* **checkpoints** - to store checkpoints after training so that they can be easily downloaded 

You can review currently available volumes under https://modal.com/unstructured-io/storage
