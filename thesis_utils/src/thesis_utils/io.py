"""I/O utilities"""
from typing import Any
import json
import pickle


def load_json(filename: str) -> dict:
    """Load a JSON file"""
    with open(filename, "r") as fp:
        d = json.load(fp)
    return d


def write_json(data: dict, filename: str) -> None:
    """Write data to a JSON file"""
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=4)


def load_pickle(filename: str) -> Any:
    """Load a pickle file"""
    with open(filename, "rb") as fp:
        obj = pickle.load(fp)
    return obj
