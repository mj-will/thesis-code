"""I/O utilities"""
from typing import Any
import json
import pickle

import h5py


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


def hdf5_to_dict(hdf5_file: h5py.File, path: str = "/") -> dict:
    """Convert an HDF5 file to a nested dictionary."""
    d = {}
    for key in hdf5_file[path]:
        if isinstance(hdf5_file[path + "/" + key], h5py.Group):
            d[key] = hdf5_to_dict(hdf5_file, path + "/" + key)
        else:
            try:
                val = hdf5_file[path + "/" + key][:]
            except ValueError:
                val = hdf5_file[path + "/" + key][()]
            if isinstance(val, bytes):
                val = val.decode()
            d[key] = val
    return d


def load_hdf5(filename: str) -> dict:
    """Load an HDF5 file.

    Parameters
    ----------
    filename
        Name of HDF5 file to load.
    """
    with h5py.File(filename, "r") as hdf5_file:
        d = hdf5_to_dict(hdf5_file)
    return d
