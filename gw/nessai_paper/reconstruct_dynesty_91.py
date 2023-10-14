"""Reconstruct the corrupted file from the pickle file"""
import os
import pickle
import json

from bilby.core.prior import PriorDict, Uniform
from bilby.core.result import BilbyJsonEncoder, Result
import dynesty
import numpy as np
import pandas as pd


RECON_PATH = "reconstructed_results"
RECON_FILE = f"{RECON_PATH}/dynesty_nomarg_91.json"


def main():
    os.makedirs(RECON_PATH, exist_ok=True)

    path = "/home/michael.williams/git_repos/nessai-paper/analysis/outdir_dynesty/result/dynesty_data91_0_analysis_H1L1V1_dynesty_dynesty.pickle.backup"

    with open(path, "rb") as fp:
        data = pickle.load(fp)

    search_parameter_keys = [
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
    ]
    weights = np.exp(data.logwt - data.logz[-1])
    nested_samples = pd.DataFrame(
        data.samples,
        columns=search_parameter_keys,
    )
    nested_samples["weights"] = weights
    nested_samples["log_likelihood"] = data.logl

    posterior_samples = pd.DataFrame(
        dynesty.utils.resample_equal(data.samples, weights),
        columns=search_parameter_keys,
    )

    # Dummy priors
    priors = PriorDict()
    for key in search_parameter_keys:
        priors[key] = Uniform(0, 1)

    result = Result(
        sampler="dynesty",
        label="dynesty",
        nested_samples=nested_samples,
        posterior=posterior_samples,
        search_parameter_keys=search_parameter_keys,
        priors=priors,
    )

    result.save_to_file(RECON_FILE)

    # json_data = dict(
    #     search_parameter_keys=search_parameter_keys,
    #     nested_samples=nested_samples,
    #     posterior_samples=posterior_samples,
    # )

    # with open(RECON_FILE, "w") as fp:
    #     json.dump(json_data, fp, cls=BilbyJsonEncoder, indent=4)


if __name__ == "__main__":
    main()
