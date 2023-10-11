#!/usr/bin/env
"""
Run nessai with and without resetting the flow.

Michael J. Williams 2023
"""
import os
import sys

import matplotlib.pyplot as plt
from nessai_models import Gaussian
from nessai.flowsampler import FlowSampler
from nessai.plot import corner_plot
from nessai.utils import setup_logger
from thesis_utils.plotting import save_figure, set_plotting
from thesis_utils import colours


def main():

    reset_flow = int(sys.argv[1])

    if reset_flow == 0:
        reset_flow = False
        output = "outdir/no_reset/"
    else:
        output = f"outdir/reset_{reset_flow}/"

    logger = setup_logger(output=output)

    model = Mixture(
        models={"gaussian": 4, "uniform": 4, "halfnorm": 4, "gamma": 4}
    )

    fs = FlowSampler(
        model,
        output=output,
        constant_volume_mode=True,
        resume=False,
        seed=1234,
        reset_flow=reset_flow,
    )
    fs.run(save=True, plot=False)


if __name__ == "__main__":
    main()
