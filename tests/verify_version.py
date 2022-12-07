import sys

import super_gradients

if __name__ == "__main__":

    ci_version = sys.argv[1]
    if ci_version == super_gradients.__version__:
        sys.exit(0)
    else:
        print(f"wrong version definition:\nCI version: {ci_version}\nsuper_gradients.__version__: {super_gradients.__version__}")
        sys.exit(1)
