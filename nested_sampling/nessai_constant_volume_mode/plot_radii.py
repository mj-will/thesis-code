#!/usr/bin/env
"""
Run nessai with and without constant volume mode.

Michael J. Williams 2022
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from thesis_utils.io import load_json, load_pickle
from thesis_utils.plotting import (
    set_plotting,
    save_figure,
    get_default_figsize,
)
from thesis_utils import colours


def main():

    set_plotting()

    cvm_ns = load_pickle("outdir/cvm/nested_sampler_resume.pkl")
    no_cvm_ns = load_pickle("outdir/no_cvm/nested_sampler_resume.pkl")
    no_cvm_fuzz = load_pickle("outdir/no_cvm_fuzz/nested_sampler_resume.pkl")

    fuzz_default = no_cvm_ns._flow_proposal.fuzz
    fuzz_specified = no_cvm_fuzz._flow_proposal.fuzz

    default_radii = fuzz_default * np.array(
        no_cvm_ns.population_radii, dtype=float
    )
    fuzz_radii = fuzz_specified * np.array(
        no_cvm_fuzz.population_radii, dtype=float
    )

    rlim = np.array([3, 7])

    figsize = get_default_figsize()
    figsize[1] *= 0.33

    fig, axs = plt.subplots(
        1,
        2,
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1]},
        figsize=figsize,
    )
    fig.subplots_adjust(wspace=0, hspace=0)

    axs[0].axhline(cvm_ns.population_radii[-1], label="CVM")
    axs[0].plot(
        no_cvm_ns.population_iterations,
        default_radii,
        label=rf"No CVM - $\epsilon_{{FF}}$={fuzz_default:.3f}",
        c="C1",
        ls="--",
    )
    axs[0].plot(
        no_cvm_fuzz.population_iterations,
        fuzz_radii,
        label=rf"No CVM - $\epsilon_{{FF}}$={fuzz_specified:.1f}",
        c="C2",
        ls="-.",
    )
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel(r"$r$")

    axs[0].set_xlim(
        [0, max(no_cvm_fuzz.iteration, no_cvm_ns.iteration, cvm_ns.iteration)]
    )
    axs[0].set_ylim(rlim)

    axs[0].fill_betweenx(
        y=axs[0].get_ylim(),
        x1=0,
        x2=no_cvm_ns.training_iterations[0],
        alpha=0.25,
        zorder=-1,
        color="gray",
        lw=0.0,
    )

    hist_kwargs = dict(histtype="step", bins=np.linspace(*rlim, 16))

    axs[1].hist(
        default_radii,
        orientation="horizontal",
        color="C1",
        ls="--",
        **hist_kwargs,
    )
    axs[1].hist(
        fuzz_radii,
        orientation="horizontal",
        color="C2",
        ls="-.",
        **hist_kwargs,
    )
    axs[1].axhline(cvm_ns.population_radii[-1])
    axs[1].set_xlabel("Count")

    fig.legend(bbox_to_anchor=(0.5, -0.3), ncol=3, loc="lower center")

    save_figure(fig, "nessai_cvm_radii", "figures")


if __name__ == "__main__":
    main()
