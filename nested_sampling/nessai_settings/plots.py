#!/usr/bin/env python
import argparse
import os

os.environ["bilby_style"] = "none"

import matplotlib.pyplot as plt
from natsort import natsorted
from nessai import config as nessai_config
from nessai.plot import corner_plot
import numpy as np
import seaborn as sns

from thesis_utils.plotting import (
    set_plotting,
    save_figure,
    get_default_corner_kwargs,
    plot_acceptance,
    plot_corner_comparison,
    get_corner_figsize,
)
from thesis_utils.io import load_hdf5, load_pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=str)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--run-labels", nargs="+", type=str)
    parser.add_argument("--corner", action="store_true")
    parser.add_argument("--acceptance", action="store_true")
    return parser.parse_args()


def main():

    set_plotting()

    nessai_config.plotting.disable_style = True

    args = parse_args()

    results = []
    samplers = []
    for run in args.runs:

        results.append(load_hdf5(os.path.join(run, "result.hdf5")))
        samplers.append(
            load_pickle(os.path.join(run, "nested_sampler_resume.pkl"))
        )

    if args.corner:
        kwargs = get_default_corner_kwargs()
        kwargs["fill_contours"] = None
        kwargs["no_fill_contours"] = True
        kwargs["bins"] = 64
        kwargs["plot_datapoints"] = False
        kwargs["show_titles"] = False

        fig = plt.figure(
            figsize=0.8
            * get_corner_figsize(len(results[0]["posterior_samples"].dtype))
        )

        fig = plot_corner_comparison(
            results,
            filename=f"{args.label}_corner",
            legend_labels=args.run_labels,
            fig=fig,
            legend_position=(0.75, 0.75),
            legend_fontsize=16,
            label_kwargs=dict(fontsize=16),
            **kwargs,
        )

    if args.acceptance:
        plot_acceptance(samplers, filename=f"{args.label}_acceptance")


if __name__ == "__main__":
    main()
