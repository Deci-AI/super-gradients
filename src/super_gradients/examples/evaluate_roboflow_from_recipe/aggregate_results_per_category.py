"""
Aggregate the output of train.sh script to compute the per-category MaP.

$ aggregate_results_per_category --result_file=<path-to-result-file> --output_file=<path-to-save-aggregated-results>
"""

import json
import argparse

from super_gradients.training.datasets.detection_datasets.roboflow.metadata import DATASETS_METADATA


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--output-file", required=False, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    import pandas as pd

    args = parse_args()
    results = pd.read_csv(args.result_file, names=["dataset_name", "map"])

    metadata = pd.read_json(json.dumps(DATASETS_METADATA), orient="index").reset_index().rename(columns={"index": "dataset_name"})

    results_with_metadata = results.merge(metadata, on="dataset_name")
    results_per_category = results_with_metadata.groupby(["category"])[["map"]].mean()

    if args.output_file is None:
        print(results_per_category)
    else:
        results_per_category.to_csv(args.output_file)
