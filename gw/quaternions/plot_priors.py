#!/usr/bin/env
"""
Plot the priors for the quaterions.
"""
from dataclasses import dataclass

import bilby
import corner
import numpy as np

from quaternions import quaternion_to_source_angles
from thesis_utils.plotting import (
    get_default_corner_kwargs,
    save_figure,
    set_plotting,
)
from thesis_utils import conf


@dataclass
class AnalyticPrior:
    x: np.ndarray
    pdf: np.ndarray


def main():

    set_plotting()

    priors = {}
    for i in range(4):
        priors[f"q_{i}"] = bilby.core.prior.Gaussian(
            0, 1, name=f"q_{i}", latex_label=f"$q_{i}$"
        )
    priors = bilby.core.prior.PriorDict(priors)
    quaternion_names = list(priors.keys())

    samples = priors.sample(100_000)

    quaternions = np.array([samples[q] for q in quaternion_names]).T

    source_angles = quaternion_to_source_angles(quaternions)

    corner_kwargs = get_default_corner_kwargs()
    corner_kwargs["quantiles"] = None
    corner_kwargs["show_titles"] = False

    quaternion_labels = [rf"$q_{i}$" for i in range(4)]
    source_angles_labels = [r"$\psi$", r"$\theta_{JN}$", r"$\phi$"]

    fig = corner.corner(quaternions, labels=quaternion_labels, **corner_kwargs)
    save_figure(fig, "quaternions_prior", "figures")

    fig = corner.corner(
        source_angles, labels=source_angles_labels, **corner_kwargs
    )

    m = source_angles.shape[1]
    diagnonal_indices = np.where((np.arange(m**2) % (m + 1)) == 0)[0]

    n = 1000
    x_psi = np.linspace(0, np.pi, n)
    pdf_psi = np.ones(n) / np.pi
    x_phase = np.linspace(0, 2 * np.pi, n)
    pdf_phase = np.ones(n) / (2 * np.pi)
    x_theta_jn = np.linspace(0, np.pi, n)
    pdf_theta_jn = np.sin(x_theta_jn) / 2

    analytic_priors = [
        AnalyticPrior(x_psi, pdf_psi),
        AnalyticPrior(x_theta_jn, pdf_theta_jn),
        AnalyticPrior(x_phase, pdf_phase),
    ]

    for i, p in zip(diagnonal_indices, analytic_priors):
        fig.axes[i].plot(p.x, p.pdf, color=conf.highlight_colour)

    save_figure(fig, "source_angles_from_quaterions_prior", "figures")


if __name__ == "__main__":
    main()
