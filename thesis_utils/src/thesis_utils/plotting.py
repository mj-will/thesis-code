"""Utilities for plotting."""
from collections import namedtuple
import importlib.resources
from itertools import product
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd

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
    figure: matplotlib.figure.Figure,
    name: str,
    path: str = "figures",
    **kwargs: Any,
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


def make_pp_plot_bilby_results(
    results,
    confidence_interval=[0.68, 0.95, 0.997],
    lines=None,
    legend_fontsize="x-small",
    keys=None,
    title=True,
    confidence_interval_alpha=0.1,
    labels=None,
    height=None,
    width=None,
    include_legend=True,
    weight_list=None,
    palette="RdYlBu",
    colours=None,
    **kwargs,
):
    """
    Make a P-P plot for a set of runs with injected signals from bilby.

    Based on the function from bilby.

    Returns
    -------
    fig, pvals:
        matplotlib figure and a NamedTuple with attributes `combined_pvalue`,
        `pvalues`, and `names`.
    """

    if keys is None:
        keys = results[0].search_parameter_keys

    if weight_list is None:
        weight_list = [None] * len(results)

    credible_levels = list()
    for i, result in enumerate(results):
        credible_levels.append(
            result.get_all_injection_credible_levels(keys, weights=weight_list[i])
        )
    credible_levels = pd.DataFrame(credible_levels)

    if lines is None:
        if colours is None:
            colours = sns.color_palette(palette, n_colors=6)
        linestyles = ["-", "--", ":"]
        style = list(product(linestyles, colours))

    x_values = np.linspace(0, 1, 1001)

    N = len(credible_levels)
    default_height, default_width = plt.rcParams["figure.figsize"]
    if width is None:
        width = default_width
    if height is None:
        height = default_height
    fig, ax = plt.subplots(figsize=(height, width))

    if isinstance(confidence_interval, float):
        confidence_interval = [confidence_interval]
    if isinstance(confidence_interval_alpha, float):
        confidence_interval_alpha = [confidence_interval_alpha] * len(
            confidence_interval
        )
    elif len(confidence_interval_alpha) != len(confidence_interval):
        raise ValueError(
            "confidence_interval_alpha must have the same length as confidence_interval"
        )

    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1.0 - ci) / 2.0
        lower = stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color="grey")

    pvalues = []
    print("Key: KS-test p-value")
    for ii, key in enumerate(credible_levels):
        pp = np.array(
            [
                sum(credible_levels[key].values < xx) / len(credible_levels)
                for xx in x_values
            ]
        )
        pvalue = stats.kstest(credible_levels[key], "uniform").pvalue
        pvalues.append(pvalue)
        print("{}: {}".format(key, pvalue))

        if labels:
            try:
                name = labels[key]
            except (AttributeError, KeyError):
                name = key
        else:
            name = key
        label = "{} ({:2.3f})".format(name, pvalue)
        plt.plot(
            x_values,
            pp,
            ls=style[ii][0],
            c=style[ii][1],
            label=label,
            **kwargs,
        )

    Pvals = namedtuple("pvals", ["combined_pvalue", "pvalues", "names"])
    pvals = Pvals(
        combined_pvalue=stats.combine_pvalues(pvalues)[1],
        pvalues=pvalues,
        names=list(credible_levels.keys()),
    )
    print("Combined p-value: {}".format(pvals.combined_pvalue))

    if title:
        ax.set_title(
            "N={}, p-value={:2.4f}".format(len(credible_levels), pvals.combined_pvalue)
        )
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    if include_legend:
        ax.legend(
            handlelength=2,
            labelspacing=0.25,
            frameon=False,
            fontsize=legend_fontsize,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig, pvals


def crop_pdf(
    filename: str, top: float, bottom: float, left: float, right: float
) -> None:
    """Crop a PDF"""
    from pypdf import PdfWriter, PdfReader

    with open(filename, "rb") as f:
        orig = PdfReader(f)
        cropped = PdfWriter()

        image = orig.pages[0]
        image.mediabox.upper_right = (
            image.mediabox.right - right,
            image.mediabox.top - top,
        )
        image.mediabox.lower_left = (
            image.mediabox.left + left,
            image.mediabox.bottom + bottom,
        )
        cropped.add_page(image)

    with open(filename, "wb") as f:
        cropped.write(f)
