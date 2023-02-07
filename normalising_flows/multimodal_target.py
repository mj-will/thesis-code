#!/usr/bin/env python
"""
Example of training a normalising flow to learn a multimodal target
distribution.

Michael J. Williams 2023
"""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nessai.flowmodel import FlowModel
import numpy as np
from scipy.stats import multivariate_normal

from thesis_utils.plotting import save_figure, set_plotting
from thesis_utils.random import seed_everything


def main():

    train = int(sys.argv[1])

    seed_everything()
    set_plotting()

    output = os.path.join("outdir", "multimodal_target")

    n = 2_000

    g1 = multivariate_normal(mean=-2.5 * np.ones(2))
    g2 = multivariate_normal(mean=2.5 * np.ones(2))

    def pdf(x):
        return (g1.pdf(x) + g2.pdf(x)) / 2.0

    data = np.concatenate([g1.rvs(n // 2), g2.rvs(n // 2)], axis=0)
    np.random.shuffle(data)

    config = dict(
        annealing=True,
        model_config=dict(
            n_inputs=2,
            n_blocks=4,
            n_neurons=16,
        ),
    )

    flow = FlowModel(output=output, config=config)
    flow.initialise()

    if os.path.exists(wf := os.path.join(output, "model.pt")) and not train:
        print("Loading existing weights")
        flow.load_weights(wf)
    else:
        print("Training a new flow")
        flow.train(data)

    n_grid = 100

    x = np.linspace(-5, 5, n_grid)
    grid = np.meshgrid(x, x)
    postitions = np.vstack(list(map(np.ravel, grid))).T

    pdf_true = pdf(postitions).reshape(n_grid, n_grid)
    pdf_flow = np.exp(flow.log_prob(postitions)).reshape(n_grid, n_grid)
    pdf_latent = (
        multivariate_normal(np.zeros(2))
        .pdf(postitions)
        .reshape(n_grid, n_grid)
    )

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    plt.setp(axs.flat, aspect=1.0, adjustable="box")

    kwargs = dict(
        cmap="Blues",
    )

    axs[0].contourf(grid[0], grid[1], pdf_true, **kwargs)
    axs[0].set_xlabel(r"$x_0$")
    axs[0].set_ylabel(r"$x_1$")

    axs[1].contourf(grid[0], grid[1], pdf_latent, **kwargs)
    axs[1].set_xlabel(r"$z_0$")
    axs[1].set_ylabel(r"$z_1$")

    axs[2].contourf(grid[0], grid[1], pdf_flow, **kwargs)
    axs[2].set_xlabel(r"$x_0$")
    axs[2].set_ylabel(r"$x_1$")

    save_figure(fig, "multimodal_comparison")


if __name__ == "__main__":
    main()
