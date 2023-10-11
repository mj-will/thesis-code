#!/usr/bin/env
"""
Run nessai with and without constant volume mode.

Michael J. Williams 2022
"""
import os

import matplotlib.pyplot as plt
from nessai_models import MixtureOfDistributions
import numpy as np
import seaborn as sns
from thesis_utils.io import load_hdf5
from thesis_utils.plotting import (
    set_plotting,
    save_figure,
    get_default_figsize,
)
from thesis_utils import colours


def main():

    set_plotting()

    cvm = load_hdf5("outdir/cvm/result.hdf5")
    no_cvm = load_hdf5("outdir/no_cvm/result.hdf5")
    no_cvm_fuzz = load_hdf5("outdir/no_cvm_fuzz/result.hdf5")
    cvm_reset = load_hdf5("outdir/cvm_reset/result.hdf5")

    model = MixtureOfDistributions(
        distributions={"gaussian": 4, "uniform": 4, "halfnorm": 4, "gamma": 4}
    )

    labels = {"gaussian": r"$x_{Gaussian}$"}

    true_posterior = {}

    for name in model.names:
        x = np.linspace(model.bounds[name][0], model.bounds[name][1], 100)
        y = np.exp(model.mapping[name](x))
        # y = np.cumsum(y) / np.sum(y)
        true_posterior[name] = (x, y)

    fig, axs = plt.subplots(4, 4, sharey="row")

    # axs = axs.ravel()

    hist_kwargs = dict(
        density=True,
        # cumulative=True,
        bins=32,
        histtype="step",
        ls="--",
    )

    for ax, n in zip(axs.T, model.names):
        # hist_kwargs["bins"] = np.linspace(*model.bounds[n], 10000)
        ax.plot(true_posterior[n][0], true_posterior[n][1], color="k")
        # ax.hist(cvm["posterior_samples"][n], **hist_kwargs)
        ax.hist(no_cvm["posterior_samples"][n], **hist_kwargs)
        # ax.hist(no_cvm_fuzz["posterior_samples"][n], **hist_kwargs)
        ax.hist(cvm_reset["posterior_samples"][n], **hist_kwargs)
    fig.tight_layout()
    save_figure(fig, "nessai_cvm_comparison", "figures")


if __name__ == "__main__":
    main()
