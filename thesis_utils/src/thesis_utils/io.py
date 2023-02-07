"""I/O utilities"""
import json


def load_json(filename: str) -> dict:
    """Load a JSON file"""
    with open(filename, "r") as fp:
        d = json.load(fp)
    return d


def write_json(data: dict, filename: str) -> None:
    """Write data to a JSON file"""
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=4)
