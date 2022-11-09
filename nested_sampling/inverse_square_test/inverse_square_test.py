#!/usr/bin/env python
"""Run the inverse square test for samplers in bilby."""
import argparse
import os

import numpy as np
import bilby

SAMPLERS = dict(
    nessai=dict(
        nlive=1000,
        reset_flow=16,
        max_iteration=50_000,
    ),
    dynesty=dict(
        nlive=500,
        nact=10,
        plot=True,
    ),
)


class InverseSquareLikelihood(bilby.Likelihood):
    """Likelihood for the inverse square test in two dimensions."""

    def __init__(self):
        super().__init__(parameters={"x": None, "y": None})

    def log_likelihood(self):
        """Log-likelihood"""
        return -np.log(self.parameters["x"] ** 2 + self.parameters["y"] ** 2)


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sampler", type=str, help="Name of the sampler to test"
    )
    return parser.parse_args()


def main():

    args = parse_args()

    output = os.path.join("outdir", f"ist_{args.sampler}")
    label = "ist"
    bilby.core.utils.setup_logger(outdir=output, log_level="INFO", label=label)

    priors = dict(
        x=bilby.core.prior.Uniform(-1, 1, name="x"),
        y=bilby.core.prior.Uniform(-1, 1, name="y"),
    )

    sampler_kwargs = SAMPLERS.get(args.sampler, {})

    bilby.run_sampler(
        likelihood=InverseSquareLikelihood(),
        priors=priors,
        outdir=output,
        label=label,
        resume=False,
        sampler=args.sampler,
        clean=True,
        **sampler_kwargs,
    )


if __name__ == "__main__":
    main()
