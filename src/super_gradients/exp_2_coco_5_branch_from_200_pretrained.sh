#!/bin/bash
train_from_recipe.py --config-name=coco2017_yolo_nas_s training_hyperparams.max_epochs=300 experiment_name=coco_5_branch_from_200_pretrained next_stage_start_epoch=200 num_gpus=8 num_branches=5
