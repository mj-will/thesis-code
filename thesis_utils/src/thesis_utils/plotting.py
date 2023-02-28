"""Utilities for plotting."""
import importlib.resources
import os
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from . import conf


def set_plotting() -> None:
    """Set the plotting defaults"""
    sns.set_palette("colorblind")
    # sns.set_style("ticks")
    # sns.set_context("paper")
    with importlib.resources.path("thesis_utils.conf", "thesis.mplstyle") as p:
        plt.style.use(p)


def get_default_figsize() -> Tuple[float, float]:
    """Get the default figure size"""
    return plt.rcParams["figure.figsize"]


def get_default_corner_kwargs() -> Dict:
    """Get the default corner kwargs"""
    return conf.default_corner_kwargs.copy()


def save_figure(
    figure: matplotlib.figure.Figure, name: str, path: str = "figures"
) -> None:
    """Save a figure with correct format.

    Will create the path if it does not exist already.

    Parameters
    ----------
    figure
        The figure to save
    name
        Name for the file without the filetype
    path
        Path to the directory where the figure should be saved
    """
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, name + f".{conf.figure_ftype}")
    figure.savefig(filename)
