#!/usr/bin/env
"""
Plot the antenna beam pattern functions.

Michael J. Williams 2023
"""
import argparse
import os
import numpy as np
from antenna_patterns import f_cross, f_plus
from thesis_utils.coordinates import spherical_to_cartesian
from thesis_utils.io import load_json, write_json
from thesis_utils.plotting import get_default_figsize, crop_pdf
from typing import Optional

import plotly.graph_objects as go
import chart_studio.plotly as py


def parse_args() -> argparse.ArgumentParser:
    """Parse the arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the figure to chart studio",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Produce the HTML version",
    )
    parser.add_argument(
        "--n-points",
        default=200,
        help="Number of points in the linspace",
        type=int,
    )
    return parser.parse_args()


def upload(fig: go.Figure, filename: str) -> str:
    """Upload the figure and return the URL"""
    return py.plot(fig, filename=filename, auto_open=False)


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
            # colorscale="Bluyl",
            colorscale="deep",
            cmin=0.0,
            cmid=0.5,
            cmax=1.0,
            colorbar=dict(
                lenmode="fraction",
                len=0.5,
                thickness=20,
                orientation="h",
                y=0.05,
            ),
        ),
    )
    return fig


def main() -> None:
    figure_dir = "figures"
    html_dir = "html"

    args = parse_args()

    figsize_inches = get_default_figsize()
    dpi = 100

    # A third of the width for 3 subplots
    figsize_inches[0] /= 2.8
    figsize = dpi * figsize_inches
    figsize *= 0.8

    dist = 1.5
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=dist, y=dist, z=dist),
    )

    scene = dict(
        camera=camera,
        xaxis=dict(range=[-0.65, 0.65], nticks=5, autorange=False),
        yaxis=dict(range=[-0.65, 0.65], nticks=5, autorange=False),
        zaxis=dict(range=[-1.0, 1.0], nticks=5, autorange=False),
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False,
        aspectratio={"x": 1.0, "y": 1.0, "z": 1.5},
    )

    ifo = go.Scatter3d(
        x=[0.5, 0, 0],
        y=[0, 0, 0.5],
        z=[0, 0, 0],
        mode="lines",
        showlegend=False,
        line=dict(
            width=4,
            color="red",
        ),
    )

    n = args.n_points
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    psi = np.linspace(0, np.pi, 11, endpoint=True)

    theta_grid, phi_grid = np.meshgrid(theta, phi)

    def power_func(theta, phi, psi):
        return f_cross(theta, phi, psi) ** 2 + f_plus(theta, phi, psi) ** 2

    power = power_func(theta_grid, phi_grid, 0)
    fp = f_plus(theta_grid, phi_grid, 0) ** 2
    fc = f_cross(theta_grid, phi_grid, 0) ** 2

    x, y, z = spherical_to_cartesian(theta_grid, phi_grid, 1.0)
    plus_data = {
        "func": fp,
        "filename": "plus_antenna_pattern",
        "title": "Plus antenna pattern",
    }

    cross_data = {
        "func": fc,
        "filename": "cross_antenna_pattern",
        "title": "Cross antenna pattern",
    }
    for data in [plus_data, cross_data]:

        sf = data["func"]
        filename = data["filename"]

        fig = plot_surface(sf * x, sf * y, sf * z, surfacecolor=sf)
        fig.add_trace(ifo)

        if args.html:

            fig.update_layout(
                scene=scene,
                title=data["title"],
            )

            if args.upload:
                url = upload(fig, filename)
                url_data = load_json("urls.json")
                url_data[filename] = url
                write_json(url_data, "urls.json")
            else:
                fig.write_html(os.path.join(html_dir, filename + ".html"))

        else:
            fig.update_layout(
                autosize=False,
                font_family="Serif",
                scene=scene,
                margin_l=0,
                margin_t=0,
                margin_b=5,
                margin_r=0,
                width=figsize[0],
                height=figsize[1],
                title=data["title"],
            )

            fig.write_image(
                os.path.join(figure_dir, filename + ".pdf"),
                width=figsize[0],
                height=figsize[1],
            )

            crop_pdf(
                os.path.join(figure_dir, filename + ".pdf"),
                30,
                10,
                0,
                0,
            )


if __name__ == "__main__":
    main()
