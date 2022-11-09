"""I/O utilities"""
import json


def load_json(filename: str) -> dict:
    """Load a JSON file"""
    with open(filename, "r") as fp:
        d = json.load(fp)
    return d
