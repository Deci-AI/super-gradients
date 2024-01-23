import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with open(path, "r") as file:
        data = json.load(file)
    return data


def dump_json(path: Path, data: Any) -> None:
    with open(path, "w") as file:
        json.dump(data, file)
