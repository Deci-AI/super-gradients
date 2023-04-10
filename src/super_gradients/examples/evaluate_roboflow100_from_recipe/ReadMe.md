# How to evaluate on Roboflow100
This is a quick tutorial on how to evaluate multiple datasets from [roboflow100](https://blog.roboflow.com/roboflow-100/).
If you want to evaluate a single dataset you should rather directly work with `train_from_recipe.py`.

### 0. Define a recipe
SuperGradients provides 2 recipes `roboflow_ppyoloe.yaml` or `roboflow_yolox.yaml`.
If you want to work with your own recipe, just make sure to include at least 2 variables in your config file:
- dataset_name: Name of the [roboflow100 dataset](https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv) to run. 
- result_path: Where the result of the experiment will be saved. Should be a full path.

These parameters are required by the script to evaluate on multiple datasets and save the results. (See section below)

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

Note that the `<output-file-name>'` will be saved in the same folder as the `train.sh` script.

### 3. Aggregate the results per category
Your results should automatically be aggregated in the same folder as the script, under name `aggregated_results.csv`. 

If for any reason you have the result file but don't have this aggregated version, you can always run the 
aggregation step manually.

- You can aggregate the results per category using this command: `python aggregate_results_per_category.py --result_file=<path-to-result-file> --output_file=<path-to-save-aggregated-results>`
- You can also just print the results by skipping the `--output_file`.

Example: `python aggregate_results_per_category.py --result_file=results.csv`
