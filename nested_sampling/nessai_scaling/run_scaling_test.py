#!/usr/bin/env python
import argparse
import os
import time
from timeit import default_timer as timer
from typing import Sequence, Union

from nessai.flowsampler import FlowSampler
from nessai.utils import setup_logger
from nessai_models import Gaussian
import numpy as np


class ModelWithDelay(Gaussian):
    def __init__(
        self,
        delay: float,
        dims: int = 2,
        bounds: Union[Sequence[float], np.ndarray] = [-10, 10],
        mean: Union[Sequence[float], np.ndarray] = None,
        cov: Union[Sequence[float], np.ndarray] = None,
        normalise: bool = False,
    ) -> None:
        super().__init__(dims, bounds, mean, cov, normalise)
        self.delay = 1e-3 * delay

    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        time.sleep(self.delay)
        return super().log_likelihood(x)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, help="Delay in ms")
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--npool", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=str, default="outdir")
    parser.add_argument("--vectorised", action="store_true")
    return parser.parse_args()


def get_output(args):
    label = f"delay_test_{args.dims}d_{args.delay}ms"
    if args.vectorised:
        label += "_vectorised"
    if args.npool:
        label += f"_npool_{args.npool}"
    output = os.path.join(args.output, label)
    return output


def main():

    args = parse_args()

    output = get_output(args)

    logger = setup_logger(output=output, log_level="INFO")

    logger.info(f"Output directory: {output}")
    logger.info(f"Delay={args.delay} ms")

    model = ModelWithDelay(args.delay, dims=args.dims)

    model.allow_vectorised = args.vectorised

    start = timer()

    fs = FlowSampler(
        model,
        output=output,
        n_pool=args.npool,
        log_on_iteration=False,
        logging_interval=30,
        resume=False,
        plot=False,
        seed=args.seed,
        reset_flow=4,
    )

    fs.run(plot=False)

    end = timer()

    elapsed = end - start

    time_file = os.path.join(output, "time.txt")

    with open(time_file, "w") as f:
        f.write(str(elapsed))


if __name__ == "__main__":
    main()
