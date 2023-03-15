import argparse
from super_gradients.training.datasets.detection_datasets.roboflow.utils import get_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", default=None)
    args = parser.parse_args()

    datasets = get_datasets(args.categories)
    print("\n".join(datasets))
