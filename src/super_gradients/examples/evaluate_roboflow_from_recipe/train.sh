#!/bin/bash
# Note: tweeter-profile seems to be lacking data so it was removed

counter=0
datasets=$(pwd)/../../training/datasets/detection_datasets/roboflow/datasets_metadata.csv


awk -F "," '{print $1}' $datasets | tail -n +2 | while read x; do
    echo "\n\n\n> [${counter}/100] ${x}\n\n"
    /home/louis.dupont/.conda/envs/louis-dev/bin/python3 -u /home/louis.dupont/PycharmProjects/super-gradients/src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=roboflow_yolox dataset_name=$x
    counter=$((counter + 1))
done
