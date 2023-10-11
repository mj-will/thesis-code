#!/usr/bin/env python
import argparse
import os
import bilby
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from thesis_utils.gw.quaternions import generate_all_bbh_parameters
from thesis_utils.gw import get_cbc_parameter_labels
from thesis_utils.plotting import (
    set_plotting,
    get_corner_figsize,
    get_default_corner_kwargs,
    plot_multiple_bilby,
    save_figure,
)
import thesis_utils.colours as thesis_colours


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot corner plots from results files"
    )
    parser.add_argument(
        "-r", "--results", nargs="+", help="List of results files to use."
    )
    parser.add_argument(
        "-f", "--filename", default=None, help="Output file name."
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        default=None,
        help="List of labels to use for each result.",
    )
    parser.add_argument(
        "-p",
        "--parameters",
        nargs="+",
        default=None,
        help="List of parameters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_plotting()

    results = [
        bilby.core.result.read_in_result(filename=r) for r in args.results
    ]

    for r in results:
        r.posterior = generate_all_bbh_parameters(r.posterior)

    # bilby.core.result.plot_multiple(
    #     results, filename=args.filename,
    #     labels=args.labels,
    #     parameters=args.parameters,
    #     corner_labels=get_cbc_parameter_labels(args.parameters),
    # )

    corner_kwargs = get_default_corner_kwargs()
    corner_kwargs.pop("fill_contours")
    corner_kwargs["show_titles"] = False
    corner_kwargs["no_fill_contours"] = True
    corner_kwargs["plot_datapoints"] = False
    corner_kwargs.pop("color")

    parameters = args.parameters
    colours = [thesis_colours.teal, thesis_colours.yellow]
    labels = args.labels

    fig = plot_multiple_bilby(
        results,
        parameters=parameters,
        labels=labels,
        colours=colours,
        corner_labels=get_cbc_parameter_labels(parameters, units=True),
        fig=plt.figure(figsize=0.85 * get_corner_figsize(len(parameters))),
        add_legend=False,
        labelpad=0.1,
        **corner_kwargs,
    )

    handles = []
    for l, c in zip(labels, colours):
        handles.append(Line2D([0], [0], color=c, label=l))

    fig.legend(
        handles=handles, loc="center", fontsize=14, bbox_to_anchor=(0.75, 0.75)
    )
    save_figure(fig, args.filename, "figures")


if __name__ == "__main__":
    main()
