#!/usr/bin/env python
"""Produce a 3-d surface plot for a likelihood."""
import argparse

import nessai
from nessai.livepoint import numpy_array_to_live_points
import nessai_models as nm
import numpy as np

import chart_studio.plotly as py
import plotly.graph_objects as go

from thesis_utils.io import load_json, write_json
from thesis_utils.random import seed_everything

seed_everything()


def parse_args() -> argparse.ArgumentParser:
    """Parse the arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Name of the model")
    parser.add_argument(
        "--n-points", default=64, help="Number of points in the linspace"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the figure to chart studio",
    )
    return parser.parse_args()


def plot_surface(model: nessai.model.Model, n_points: int = 64) -> go.Figure:
    """Produce the surface plot"""

    if model.dims != 2:
        raise ValueError("Can only plot two-dimensional models!")

    x, y = np.linspace(model.lower_bounds, model.upper_bounds, n_points).T
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)

    points = np.concatenate([xx, yy], axis=1)
    points = numpy_array_to_live_points(points, model.names)
    log_likelihood = model.log_likelihood(points)

    surface = go.Surface(
        z=log_likelihood.reshape(n_points, n_points),
        x=x,
        y=y,
        colorscale="Plasma",
        colorbar=dict(title="Log-likelihood"),
    )

    fig = go.Figure(
        data=[
            surface,
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title=r"x<sub>0</sub>",
            yaxis_title=r"x<sub>1</sub>",
            zaxis_title=r"Log-likelihood",
        ),
    )
    return fig


def get_model_class(name: str) -> nessai.model.Model:
    """Get the nessai model class"""
    return getattr(nm, name)


def upload(fig: go.Figure, filename: str) -> str:
    """Upload the figure and return the URL"""
    return py.plot(fig, filename=filename, auto_open=False)


def main() -> None:
    """Produce the surface plots."""
    args = parse_args()

    Model = get_model_class(args.model)
    model = Model(dims=2)

    fig = plot_surface(model, n_points=args.n_points)
    fig.update_layout(title=f"{args.model} log-likelihood")

    if args.upload:
        filename = f"{args.model.lower()}_log_likelihood"
        url = upload(fig, filename)

        url_data = load_json("urls.json")
        url_data[filename] = url
        write_json(url_data, "urls.json")

    else:
        fig.show()


if __name__ == "__main__":
    main()
