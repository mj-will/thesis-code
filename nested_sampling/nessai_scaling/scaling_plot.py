#!/usr/bin/env python
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from thesis_utils import conf
from thesis_utils.plotting import save_figure, set_plotting
from thesis_utils.sorting import natural_sort
from thesis_utils.io import load_json

from run_scaling_test import get_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, help="Delay in ms", nargs="+")
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--output", type=str, default="figures")
    parser.add_argument("--resultsdir", type=str, default="outdir")
    return parser.parse_args()


def amdahls_law(p, n_cores):
    """Compute Amdahl's law from a given proportion p"""
    if p > 1:
        raise ValueError("p must be less than 1")
    return 1 / ((1 - p) + p / n_cores)


def main():
    # conf.figure_ftype = "png"
    set_plotting()
    args = parse_args()

    args.vectorised = None
    args.npool = None

    results_per_delay = {}
    fraction_per_delay = {}

    for delay in args.delay:
        print(f"Delay = {delay}")
        runs_path = (
            f"{args.resultsdir}/delay_test_{args.dims}d_{delay}ms_npool"
        )
        print(runs_path)
        runs = natural_sort(glob.glob(runs_path + "*"))
        results = {}

        for rp in runs:
            npool = re.search(r"npool_\d*", rp)
            if npool is None:
                npool = 0
            else:
                npool = int(npool.group().split("_")[-1])
            try:
                time = np.genfromtxt(os.path.join(rp, "time.txt"))
            except FileNotFoundError:
                print(f"Skipping {npool}")
                continue
            results[npool] = time
            if npool == 1:
                res = load_json(os.path.join(rp, "result.json"))
                fraction_per_delay[delay] = (
                    res["likelihood_evaluation_time"] / time
                )

        results_per_delay[delay] = results

    print(fraction_per_delay)

    fig = plt.figure()

    optimal = np.arange(1, 40, 1)
    plt.plot(optimal, optimal, color="k", label="Optimal")
    markers = ["o", "+", "^", "x", "s", "*"]
    colours = sns.color_palette(n_colors=len(results_per_delay))

    for i, (delay, results) in enumerate(results_per_delay.items()):
        print(f"Plotting delay = {delay}")
        pool_vec = np.fromiter(results.keys(), int)
        timings = np.fromiter(results.values(), float)

        if pool_vec[0] != 1:
            print("Initial n_pool is not 1!")

        speedup = timings[0] / timings

        plt.plot(
            pool_vec,
            speedup,
            ls="",
            marker=markers[i],
            color=colours[i],
            label=f"{delay} ms",
        )
        plt.plot(
            optimal,
            amdahls_law(fraction_per_delay[delay], optimal),
            ls="--",
            color=colours[i],
        )

    plt.legend()
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    save_figure(fig, f"scaling_{args.dims}d_{args.delay}ms", args.output)


if __name__ == "__main__":
    main()
