#!/bin/bash
# Note: tweeter-profile seems to be lacking data so it was removed

counter=0
datasets=$(pwd)/datasets.txt
configname="$1"

awk -F "," '{print $1}' $datasets | tail -n +2 | while read dataset_name; do
    echo "\n\n\n> [${counter}/100] ${dataset_name}\n\n"
    python -u ../train_from_recipe_example/train_from_recipe.py --config-name=$configname dataset_name=$dataset_name
    counter=$((counter + 1))
done
