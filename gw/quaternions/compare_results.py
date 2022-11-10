# /usr/bin/env python
import argparse
import os

import bilby

from thesis_utils.gw import get_cbc_parameter_labels
from thesis_utils.plotting import set_plotting


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=str)
    parser.add_argument("--filename", type=str, default="comparison.png")
    parser.add_argument("--labels", nargs="+", type=str, default=None)
    return parser.parse_args()


def main():

    set_plotting()

    os.environ["bilby_style"] = "none"
    args = parse_args()

    results = []
    for result_file in args.results:
        results.append(bilby.core.result.read_in_result(result_file))

    parameters = results[0].search_parameter_keys
    if "q_0" in parameters:
        for i in range(4):
            parameters.remove(f"q_{i}")
        parameters += ["psi", "theta_jn", "phase"]

    bilby.core.result.plot_multiple(
        results,
        filename=args.filename,
        labels=args.labels,
        parameters=parameters,
        fill_contours=False,
        corner_labels=get_cbc_parameter_labels(parameters),
    )


if __name__ == "__main__":
    main()
