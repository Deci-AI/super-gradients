# How to evaluate on Roboflow100

### 0. Define a recipe
Create a recipe similar to `roboflow_ppyoloe.yaml` or `roboflow_yolox.yaml`.
This recipe require to include 2 parameters:
- dataset_name: Name of the [roboflow100 dataset](https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv) to run. 
- result_path: Where the result of the experiment will be saved. Should be a full path.

### 1. Chose your datasets
List all the datasets that you want to evaluate on in the `datasets.txt` file
Example:
```
aerial-pool
secondary-chains
aerial-spheres
soccer-players-5fuqs
weed-crop-aerial
aerial-cows
cloud-types
```

### 2. Launch the training/evaluation script.
You can launch the command like this: `sh train.sh <config-name> <output-file-name>`
Example: `sh train.sh roboflow_ppyoloe results.csv`
Note that the `'output-file-name'` will be saved in the same folder as the `train.sh` script

### 3. Aggregate the results
You can aggregate the results per category using this command: `python aggregate_results_per_category.py --result_file=<path-to-result-file> --output_file=<path-to-save-aggregated-results>`
You can also just print the results by skiping the `--output_file`.

Example: `python aggregate_results_per_category.py --result_file=results.csv`
