#!/usr/bin/env
"""
Plot the antenna beam pattern functions.

Michael J. Williams 2023
"""
import os
import numpy as np
from antenna_patterns import f_cross, f_plus
from thesis_utils.coordinates import spherical_to_cartesian
from thesis_utils.plotting import get_default_figsize
from typing import Optional

import plotly.graph_objects as go


def plot_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    surfacecolor: Optional[np.ndarray] = None,
    layout: Optional[go.Layout] = None,
    fig: Optional[go.Figure] = None,
) -> go.Figure:
    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surfacecolor,
            colorscale="tealgrn",
            cmin=0.0,
            cmax=1.0,
            colorbar=dict(
                lenmode="fraction",
                len=0.5,
                thickness=20,
                orientation="h",
                y=-0.1,
            ),
        ),
    )
    return fig


def main() -> None:
    figure_dir = "figures"

    figsize_inches = get_default_figsize()
    dpi = 100

    # A third of the width for 3 subplots
    figsize_inches[0] /= 2
    figsize = dpi * figsize_inches

    dist = 1.8
    camera = dict(
        # up=dict(x=0, y=0, z=1),
        # center=dict(x=0, y=0, z=0),
        eye=dict(x=dist, y=dist, z=dist)
    )

    scene = dict(
        camera=camera,
        xaxis=dict(range=[-0.65, 0.65], nticks=5),
        yaxis=dict(range=[-0.65, 0.65], nticks=5),
        zaxis=dict(range=[-1.0, 1.0], nticks=5),
    )

    ifo = go.Scatter3d(
        x=[0.6, 0, 0],
        y=[0, 0, 0.6],
        z=[0, 0, 0],
        mode="lines",
        showlegend=False,
        line=dict(
            width=4,
            color="red",
        ),
    )

    n = 100
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    psi = np.linspace(0, np.pi, 11, endpoint=True)

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    def power_func(theta, phi, psi):
        return f_cross(theta, phi, psi) ** 2 + f_plus(theta, phi, psi) ** 2

    power = power_func(theta_grid, phi_grid, 0)
    fp = f_plus(theta_grid, phi_grid, 0) ** 2
    fc = f_cross(theta_grid, phi_grid, 0) ** 2

    x, y, z = spherical_to_cartesian(theta_grid, phi_grid, fc)
    fig = plot_surface(x, y, z, surfacecolor=fc)
    fig.add_trace(ifo)

    fig.update_layout(
        font_family="Serif",
        scene=scene,
        margin_l=5,
        margin_t=5,
        margin_b=5,
        margin_r=5,
    )

    fig.write_image(
        os.path.join(figure_dir, "antenna_cross.pdf"),
        width=figsize[0],
        height=figsize[1],
    )

    x, y, z = spherical_to_cartesian(theta_grid, phi_grid, fp)
    fig = plot_surface(x, y, z, surfacecolor=fp)
    fig.add_trace(ifo)

    fig.update_layout(
        font_family="Serif",
        scene=scene,
        margin_l=5,
        margin_t=5,
        margin_b=5,
        margin_r=5,
    )

    fig.write_image(
        os.path.join(figure_dir, "antenna_plus.pdf"),
        width=figsize[0],
        height=figsize[1],
    )


if __name__ == "__main__":
    main()
