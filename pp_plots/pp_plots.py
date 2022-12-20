#!/usr/bin/env python
"""Comparison of PP-plots.

Michael J. Williams 2022
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from thesis_utils.plotting import (
    save_figure,
    set_plotting,
)
from thesis_utils.random import seed_everything


def main():

    seed_everything()
    set_plotting()

    n = 100
    n_post = 5000
    n_steps = 1000

    examples = dict(
        over=dict(post_scale=5.0, sample_scale=2.5, label="Over-constrained"),
        under=dict(
            post_scale=5.0, sample_scale=10.0, label="Under-constrained"
        ),
        left=dict(sample_shift=-2.0, label="Left-biased"),
        right=dict(sample_shift=2.0, label="Right-biased"),
    )

    def get_pp_plot_data(
        post_scale=5.0,
        sample_scale=5.0,
        sample_shift=0.0,
        n=100,
        n_post=5000,
        n_steps=1000,
        **kwargs,
    ):
        theta_true = 10 * np.random.randn(n)
        post = (
            theta_true[:, None] + np.random.randn(n, n_post) + np.random.rand()
        )
        u = np.random.rand(n, 1)
        post_mean = stats.norm(scale=post_scale).ppf(u)
        post = (
            theta_true[:, None]
            + post_mean
            + np.random.normal(
                loc=sample_shift, scale=sample_scale, size=(n, n_post)
            )
        )
        rho = np.sum(post < theta_true[:, None], axis=1) / n_post
        x_values = np.linspace(0, 1, n_steps)
        pp = np.array([np.sum(rho < x) / n for x in x_values])
        return pp

    confidence_intervals = [0.68, 0.95, 0.997]
    x_values = np.linspace(0, 1, n_steps, endpoint=True)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    axs = axs.ravel()
    for ci in confidence_intervals:
        edge_of_bound = (1.0 - ci) / 2.0
        lower = stats.binom.ppf(1 - edge_of_bound, n, x_values) / n
        upper = stats.binom.ppf(edge_of_bound, n, x_values) / n
        lower[0] = 0
        upper[0] = 0
        lower[-1] = 1
        upper[-1] = 1
        for ax in axs:
            ax.fill_between(x_values, lower, upper, alpha=0.1, color="k")

    for ax, example in zip(axs, examples.values()):
        pp = get_pp_plot_data(n_steps=n_steps, n=n, n_post=n_post, **example)
        ax.plot(x_values, pp)
        ax.set_title(example["label"])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    plt.tight_layout()
    save_figure(fig, "pp_plot_comparison", "figures")


if __name__ == "__main__":
    main()
