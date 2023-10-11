#!/usr/bin/env
"""
Run nessai with and without constant volume mode.

Michael J. Williams 2022
"""
import os

import matplotlib.pyplot as plt
from nessai_models import MixtureOfDistributions
from nessai.flowsampler import FlowSampler
from nessai.plot import corner_plot
from nessai.utils import setup_logger
from thesis_utils.plotting import save_figure, set_plotting
from thesis_utils import colours


def main():

    output = "outdir"

    logger = setup_logger(output=output)

    output_no_cvm = os.path.join(output, "no_cvm")
    output_no_cvm_fuzz = os.path.join(output, "no_cvm_fuzz")
    output_cvm = os.path.join(output, "cvm")
    output_cvm_98 = os.path.join(output, "cvm_98")
    output_reset = os.path.join(output, "cvm_reset")

    model = MixtureOfDistributions(
        distributions={"gaussian": 4, "uniform": 4, "halfnorm": 4, "gamma": 4}
    )

    logger.info("Running with constant volume mode")

    fs_cvm = FlowSampler(
        model,
        output=output_cvm,
        constant_volume_mode=True,
        resume=True,
        seed=1234,
    )
    fs_cvm.run(save=True, plot=False)

    model.likelihood_evaluations = 0

    fs_cvm_98 = FlowSampler(
        model,
        output=output_cvm_98,
        constant_volume_mode=True,
        volume_fraction=0.98,
        resume=True,
        seed=1234,
    )
    fs_cvm_98.run(save=True, plot=False)

    logger.info("Running without constant volume mode")
    model.likelihood_evaluations = 0
    fs_no_cvm = FlowSampler(
        model,
        output=output_no_cvm,
        constant_volume_mode=False,
        resume=True,
        seed=1234,
    )
    fs_no_cvm.run(save=True, plot=False)

    logger.info("Running without constant volume mode with fuzz")
    model.likelihood_evaluations = 0
    fs_no_cvm_fuzz = FlowSampler(
        model,
        output=output_no_cvm_fuzz,
        constant_volume_mode=False,
        resume=True,
        expansion_fraction=None,
        fuzz=1.3,
        seed=1234,
    )
    fs_no_cvm_fuzz.run(save=True, plot=False)

    model.likelihood_evaluations = 0
    fs_reset = FlowSampler(
        model,
        output=output_reset,
        constant_volume_mode=True,
        resume=True,
        seed=1234,
        reset_flow=4,
    )
    fs_reset.run(save=True, plot=False)
    set_plotting()

    # # TODO: labels

    # fig, axs = plt.subplots(4, 4, sharey="row")

    # axs = axs.ravel()

    # hist_kwargs = dict(
    #     density=True,
    #     # bins=32,
    #     histtype="step",
    # )

    # for ax, n in zip(axs, model.names):
    #     ax.hist(fs_cvm.posterior_samples[n], color=colours.teal, **hist_kwargs)
    #     ax.hist(
    #         fs_no_cvm.posterior_samples[n],
    #         color=colours.coral_pink,
    #         **hist_kwargs,
    #     )
    #     ax.hist(fs_no_cvm_fuzz.posterior_samples[n], color="C1", **hist_kwargs)
    #     ax.hist(fs_reset.posterior_samples[n], color="C3", **hist_kwargs)
    # fig.tight_layout()
    # save_figure(fig, "nessai_cvm_comparison", "figures")


if __name__ == "__main__":
    main()
