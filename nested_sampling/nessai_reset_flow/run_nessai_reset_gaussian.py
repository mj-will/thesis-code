#!/usr/bin/env python
"""
Run nessai with and without resetting the flow.

Michael J. Williams 2023
"""
import sys

from nessai_models import Gaussian
from nessai.flowsampler import FlowSampler
from nessai.utils import setup_logger


def main():

    reset_flow = int(sys.argv[1])

    if reset_flow == 0:
        reset_flow = False
        output = "outdir/no_reset_v3/"
    else:
        output = f"outdir/reset_{reset_flow}_v3/"

    setup_logger(output=output)

    model = Gaussian(50)

    fs = FlowSampler(
        model,
        output=output,
        constant_volume_mode=True,
        resume=True,
        seed=1234,
        reset_flow=reset_flow,
        flow_config=dict(n_blocks=2, model_config=dict(n_neurons="half")),
    )
    fs.run(save=True, plot=True)


if __name__ == "__main__":
    main()
