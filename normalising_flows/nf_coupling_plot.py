"""Produce a figure to show the stages of a coupling transform-based flow."""
import os
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from nessai.flowmodel import FlowModel
from nessai.livepoint import live_points_to_array
from nessai_models import Rosenbrock
import numpy as np
import seaborn as sns
import torch

from thesis_utils.plotting import (
    get_default_figsize,
    save_figure,
    set_plotting,
)
from thesis_utils.random import seed_everything
from thesis_utils.sampling.rejection import rejection_sample_model
import thesis_utils.colours as thesis_colours


def plot_transforms(
    x: np.ndarray,
    flow: FlowModel,
    arrow_colours: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """
    Plot the transformation of the samples after each transform in the flow
    """
    outputs = []
    inputs = (
        torch.from_numpy(x).type(torch.get_default_dtype()).to(flow.device)
    )

    n = 0
    for module in flow._modules["_transform"]._modules["_transforms"]:
        with torch.inference_mode():
            inputs, _ = module(inputs)
            outputs.append(inputs.cpu().numpy())
        n += 1
    print(f"Flow has {n} transforms")

    final = outputs[-1]
    pospos = np.where(np.all(final >= 0, axis=1))
    negneg = np.where(np.all(final < 0, axis=1))
    posneg = np.where((final[:, 0] >= 0) & (final[:, 1] < 0))
    negpos = np.where((final[:, 0] < 0) & (final[:, 1] >= 0))

    points = [pospos, negneg, posneg, negpos]
    colours = sns.color_palette("mako", n_colors=4)
    print("Plotting the transforms")
    print("Plotting inputs")

    with sns.axes_style(
        {
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.bottom": True,
            "xtick.top": True,
            "ytick.left": True,
            "ytick.right": True,
        }
    ):

        figsize = get_default_figsize()

        fig, ax = plt.subplots(
            1,
            n + 1,
            sharex=True,
            sharey=True,
            gridspec_kw=dict(hspace=0, wspace=0),
            figsize=(figsize[0], 0.5 * figsize[1]),
        )
        ax = ax.ravel()
        for j, c in zip(points, colours):
            ax[0].scatter(x[j, 0], x[j, 1], color=c, s=1.0)

        for i, o in enumerate(outputs, start=1):
            print(f"Plotting outputs from transform {i}")
            for j, c in zip(points, colours):
                ax[i].scatter(o[j, 0], o[j, 1], color=c, s=1.0)
            # ax[i].set_title(f"Transform {i}")

        if arrow_colours is not None:
            for i in range(n):

                arrowlen = 0.125

                xyA = [1 - arrowlen, 0.5]
                # ax[i].plot(*xyA, "o")
                xyB = [0 + arrowlen, 0.5]
                # ax[i + 1].plot(*xyB, "o")
                # ConnectionPatch handles the transform internally so no need to get fig.transFigure
                arrow = patches.ConnectionPatch(
                    xyA,
                    xyB,
                    coordsA=ax[i].transAxes,
                    coordsB=ax[i + 1].transAxes,
                    # Default shrink parameter is 0 so can be omitted
                    color=arrow_colours[i],
                    arrowstyle="-|>",  # "normal" arrow
                    mutation_scale=10,  # controls arrow head size
                    linewidth=2.0,
                )
                fig.patches.append(arrow)

        ax[0].set_title("Input")
        ax[-1].set_title("Output")
        plt.tight_layout()
    return fig


def main() -> None:
    """Training a flow and produce the transforms plot"""

    set_plotting()
    seed_everything()

    outdir = "outdir/nf_coupling_plot/"
    os.makedirs(outdir, exist_ok=True)

    model = Rosenbrock()
    data = rejection_sample_model(model, 10_000)

    config = dict(
        annealing=True,
        model_config=dict(
            n_inputs=model.dims,
            n_blocks=2,
            n_neurons=64,
            kwargs=dict(
                linear_transform=None,
                batch_norm_between_layers=True,
            ),
        ),
    )

    flow = FlowModel(output=outdir, config=config)
    flow.initialise()
    print(flow.model)

    data_unstruct = live_points_to_array(data, names=model.names)

    fig = plt.figure()
    plt.scatter(data_unstruct[:, 0], data_unstruct[:, 1])
    fig.savefig(os.path.join(outdir, "training_data.png"))

    flow.train(data_unstruct)

    fig = plot_transforms(
        data_unstruct[:2000],
        flow.model,
        arrow_colours=[
            thesis_colours.pillarbox,
            thesis_colours.yellow,
            thesis_colours.pillarbox,
            thesis_colours.yellow,
        ],
    )

    save_figure(fig, "coupling_breakdown", "figures")


if __name__ == "__main__":
    main()
