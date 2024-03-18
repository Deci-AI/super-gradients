import csv
import json
from pathlib import Path
from typing import Any
import sys


def load_txt(path: Path) -> str:
    with open(path, "r") as file:
        return file.readlines()


def load_json(path: Path) -> Any:
    with open(path, "r") as file:
        data = json.load(file)
    return data


def dump_json(path: Path, data: Any) -> None:
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def load_txt_with_json(path: Path) -> Any:
    with open(path, "r") as file:
        data = [json.loads(line) for line in file.readlines()]
    return data


def load_csv_with_json(path):
    # Set a larger field size limit
    csv.field_size_limit(sys.maxsize)

    def custom_json_loader(value):
        # Remove outer double quotes before parsing JSON
        value = value.strip('"')
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        # Create a CSV DictReader object
        csv_reader = csv.DictReader(file, quoting=csv.QUOTE_NONNUMERIC)

        # Skip the header row
        next(csv_reader, None)

        # Read the contents into a list of dictionaries
        data = []
        for row in csv_reader:
            # Apply custom_json_loader to specific columns containing JSON data
            for key, value in row.items():
                row[key] = custom_json_loader(value)

            data.append(row)
    return data
