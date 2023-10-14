"""Compute the JSD

Note this code is not optimized and may take several hours to run.
"""
import os
import sys

import numpy as np
import tqdm

from thesis_utils.io import load_json, write_json
from thesis_utils import js


try:
    import ujson as json
except ImportError:
    print("Falling back to standard json library")
    import json


def load_posterior(filename):
    with open(filename, "r") as fp:
        result = json.load(fp)
    post = result["posterior"]["content"]
    return post


SEARCH_PARAMETERS = dict(
    marg=[
        "chirp_mass",
        "mass_ratio",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "dec",
        "ra",
        "theta_jn",
        "psi",
        "geocent_time",
    ],
    nomarg=[
        "chirp_mass",
        "mass_ratio",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "luminosity_distance",
        "dec",
        "ra",
        "theta_jn",
        "psi",
        "geocent_time",
    ],
)


def main(label):
    os.makedirs("results", exist_ok=True)

    result_files = load_json("result_files_lookup.json")

    posteriors = dict()

    samplers = list(result_files.keys())

    for sampler, data in result_files.items():
        print(f"Loading posteriors for: {sampler}")
        posteriors[sampler] = dict()
        for key, file in tqdm.tqdm(data[label].items()):
            posteriors[sampler][key] = load_posterior(file)

    n = len(posteriors[samplers[0]])

    print("Computing JSD")
    parameters = SEARCH_PARAMETERS.get(label)
    overall_results = dict()
    for i in tqdm.tqdm(range(n)):
        post0 = posteriors[samplers[0]][str(i)]
        post1 = posteriors[samplers[1]][str(i)]
        results = dict()
        for key in parameters:
            results[key] = js.calculate_js(post0[key], post1[key], base=2)
        overall_results[i] = results

    write_json(overall_results, f"results/jsd_results_{label}.json")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise RuntimeError("Must specify label. Usage: plot_jsd.py nomarg")
    label = sys.argv[1]
    main(label)
