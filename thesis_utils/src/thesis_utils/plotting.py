"""Utilities for plotting."""
from collections import namedtuple
import importlib.resources
from itertools import product
import os
import socket
from typing import Any, Dict, List, Optional, Tuple, Union

import corner
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from natsort import natsorted
from nessai import config as nessai_config
from nessai.plot import corner_plot
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd


from . import conf


def set_plotting() -> None:
    """Set the plotting defaults"""
    sns.set_palette("colorblind")
    os.environ["BILBY_STYLE"] = "none"
    os.environ["bilby_style"] = "none"
    nessai_config.plotting.disable_style = True
    with importlib.resources.path("thesis_utils.conf", "thesis.mplstyle") as p:
        plt.style.use(p)

    # If using HAWK, set the latex path
    if socket.gethostname() == "cl8":
        os.environ["PATH"] = os.pathsep.join(
            ("/usr/local/texlive/2023/bin/x86_64-linux", os.environ["PATH"])
        )


def get_default_figsize() -> np.ndarray[float, float]:
    """Get the default figure size"""
    return np.array(plt.rcParams["figure.figsize"])


def get_default_corner_kwargs() -> Dict:
    """Get the default corner kwargs"""
    return conf.default_corner_kwargs.copy()


def get_corner_figsize(n: int) -> float:
    """Get the size of a corner plot based on the number of parameters"""
    figsize = get_default_figsize()
    figsize[0] = figsize[1]
    figsize *= float(n) / 3
    return figsize


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


def plot_acceptance(
    samplers: List, figsize: Tuple[float] = None, filename: Optional[str] = None
):
    """Plot the acceptance of a nested sampling run"""
    ls = nessai_config.plotting.line_styles

    if figsize is None:
        figsize = get_default_figsize()
        figsize[1] *= 1.3

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=figsize)

    for i, ns in enumerate(samplers):
        it = (np.arange(len(ns.min_likelihood))) * (ns.nlive // 10)
        dtrain = np.array(ns.training_iterations[1:]) - np.array(
            ns.training_iterations[:-1]
        )
        axs[0].plot(it, ns.mean_acceptance_history, ls=ls[i])
        axs[1].plot(
            ns.training_iterations, np.arange(len(ns.training_iterations)), ls=ls[i]
        )
        axs[2].plot(ns.training_iterations[1:], dtrain, ls=ls[i])

        axs[3].plot(ns.population_iterations, ns.population_acceptance, ls=ls[i])

    axs[0].set_ylim([0, 1])
    axs[0].set_ylabel("Acceptance")

    axs[1].set_ylabel("Cumulative \ntraining count")

    axs[2].set_ylabel("Iterations \nbetween training")

    axs[3].set_yscale("log")
    axs[3].set_ylabel("Rejection sampling \nacceptance")

    for ax in axs:
        ylims = ax.get_ylim()
        ax.set_ylim(ylims)
        ax.fill_betweenx(
            y=ax.get_ylim(),
            x1=0,
            x2=ns.training_iterations[0],
            alpha=0.25,
            zorder=-1,
            color="gray",
            lw=0.0,
        )
        ax.set_xlim([0, 75_000])

    axs[-1].set_xlabel("Iteration")

    axs[-1].legend(ncol=len(samplers), loc="center", bbox_to_anchor=(0.5, -0.4))

    if filename is not None:
        save_figure(fig, filename)
    else:
        return fig


def plot_corner_comparison(
    results,
    filename: Optional[str] = None,
    parameters: Optional[List[str]] = None,
    legend_labels: Optional[List[str]] = None,
    legend_position: Tuple[float] = None,
    legend_fontsize: int = 14,
    **kwargs,
):
    """Plot a corner plot comparing different results."""
    kwargs = kwargs.copy()
    kwargs.pop("color")
    if "hist_kwargs" not in kwargs:
        kwargs["hist_kwargs"] = {}
    colours = sns.color_palette("colorblind")
    kwargs["hist_kwargs"]["color"] = colours[0]

    if parameters is None:
        parameters = natsorted(
            set(results[0]["posterior_samples"].dtype.names)
            - set(nessai_config.livepoints.non_sampling_parameters)
        )

    labels = [rf"${p}$" for p in parameters]

    fig = kwargs.pop("fig", None)
    for i, result in enumerate(results):
        kwargs["hist_kwargs"]["color"] = colours[i]
        fig = corner_plot(
            result["posterior_samples"],
            include=parameters,
            labels=labels,
            fig=fig,
            color=colours[i],
            **kwargs,
        )

    if legend_labels is not None:
        fig.legend(
            handles=[
                Line2D([0], [0], color=colours[i], label=legend_labels[i])
                for i in range(len(legend_labels))
            ],
            loc="center",
            fontsize=legend_fontsize,
            bbox_to_anchor=legend_position,
        )

    if filename is not None:
        save_figure(fig, filename)
    else:
        return fig


def plot_multiple_bilby(
    results,
    labels=None,
    colours=None,
    corner_labels=None,
    add_legend=False,
    parameters=None,
    **kwargs,
):
    """Generate a corner plot overlaying two sets of results

    Based on various functions from bilby.
    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mpllines
    from bilby.core.result import sanity_check_labels

    kwargs["show_titles"] = False
    kwargs["title_quantiles"] = None
    if corner_labels is not None:
        kwargs["labels"] = corner_labels

    samples = [r.posterior[parameters].to_numpy() for r in results]

    lines = []
    fig = kwargs.pop("fig", None)
    for i, s in enumerate(samples):
        if colours:
            c = colours[i]
        else:
            c = "C{}".format(i)
        hist_kwargs = kwargs.get("hist_kwargs", dict())
        hist_kwargs["color"] = c
        fig = corner.corner(
            s,
            fig=fig,
            save=False,
            color=c,
            **kwargs,
        )
        lines.append(mpllines.Line2D([0], [0], color=c, label=labels[i]))

    # Rescale the axes
    for i, ax in enumerate(fig.axes):
        ax.autoscale()
    plt.draw()

    labels = sanity_check_labels(labels)

    if add_legend:
        axes = fig.get_axes()
        ndim = int(np.sqrt(len(axes)))
        axes[ndim - 1].legend(handles=lines)

    return fig


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
