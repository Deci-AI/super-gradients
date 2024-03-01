import json
from pathlib import Path
from typing import Any


def load_txt(path: Path) -> str:
    with open(path, "r") as file:
        return file.readlines()


def load_json(path: Path) -> Any:
    with open(path, "r") as file:
        data = json.load(file)
    return data


def load_json_by_line(path: Path) -> Any:
    with open(path, "r") as file:
        data = [json.loads(line) for line in file.readlines()]
    return data


def dump_json(path: Path, data: Any) -> None:
    with open(path, "w") as file:
        json.dump(data, file)
