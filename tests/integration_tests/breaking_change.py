import sys
import argparse
import json

from super_gradients.common.code_analysis.breaking_changes_detection import analyze_breaking_changes


def main():
    parser = argparse.ArgumentParser(description="Example script using flags")
    parser.add_argument("--silent", action="store_true", help="Enable verbose mode")
    parser.add_argument("--output-file", default=None, type=str, help="Output file name")
    parser.add_argument("--fail-on-error", action="store_true", help="Fail on error")

    args = parser.parse_args()

    breaking_changes_list = analyze_breaking_changes(verbose=not args.silent)

    if args.output_file:
        with open(args.output_file, "w") as file:
            json.dump(breaking_changes_list, file)

    if len(breaking_changes_list) > 0 and args.fail_on_error:
        sys.exit(2)


if __name__ == "__main__":
    main()
