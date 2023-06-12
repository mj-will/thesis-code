"""Utilities for plotting."""
import importlib.resources
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

from . import conf


def set_plotting() -> None:
    """Set the plotting defaults"""
    sns.set_palette("colorblind")
    # sns.set_style("ticks")
    # sns.set_context("paper")
    os.environ["BILBY_STYLE"] = "none"
    os.environ["bilby_style"] = "none"
    with importlib.resources.path("thesis_utils.conf", "thesis.mplstyle") as p:
        plt.style.use(p)


def get_default_figsize() -> np.ndarray[float, float]:
    """Get the default figure size"""
    return np.array(plt.rcParams["figure.figsize"])


def get_default_corner_kwargs() -> Dict:
    """Get the default corner kwargs"""
    return conf.default_corner_kwargs.copy()


def save_figure(
    figure: matplotlib.figure.Figure, name: str, path: str = "figures", **kwargs: Any
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
    figure.savefig(filename, **kwargs)


def pp_plot(
    data: np.ndarray,
    labels: Optional[List[str]] = None,
    n_steps: int = 1000,
    confidence_intervals: Optional[List[float]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    colours: Optional[Union[List, str]] = None,
) -> matplotlib.figure.Figure:
    """Produce a probability-probability plot.

    Parameters
    ----------
    data :
        Array of samples of shape (n samples x n parameters).
    labels :
        List of labels for each parameter.
    n_steps :
        Number of steps to use for plotting the curves.
    confidence_intervals :
        Confidence intervals to plot as shaded regions.
    ax :
        Axis on which to plot the P-P plot.
    colours :
        Colour(s) to use for the line(s).

    Returns
    -------
    Figure with the P-P plot.
    """

    n = data.shape[-1]
    if data.ndim == 1:
        data = data[:, np.newaxis]

    if isinstance(labels, str):
        labels = [labels]

    if confidence_intervals is None:
        confidence_intervals = [0.68, 0.95, 0.997]

    x_values = np.linspace(0, 1, n_steps, endpoint=True)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if confidence_intervals:
        for ci in confidence_intervals:
            edge_of_bound = (1.0 - ci) / 2.0
            lower = stats.binom.ppf(1 - edge_of_bound, n, x_values) / n
            upper = stats.binom.ppf(edge_of_bound, n, x_values) / n
            lower[0] = 0
            upper[0] = 0
            lower[-1] = 1
            upper[-1] = 1
            ax.fill_between(x_values, lower, upper, alpha=0.1, color="k")

    x = np.linspace(0, 1, len(data), endpoint=True)

    if colours is None:
        colours = sns.color_palette()
    elif isinstance(colours, str):
        colours = [colours]

    for d, label, c in zip(data.T, labels, colours):
        ax.plot(x, d, label=label, c=c)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    return fig
