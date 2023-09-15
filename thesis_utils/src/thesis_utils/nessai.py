""""Utilities for running nessai"""
from typing import Any, Optional

from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
import nessai_models


def get_model_from_name(name: str, **kwargs: Any) -> Model:
    if name.lower() in ["gaussian", "normal"]:
        ModelClass = nessai_models.Gaussian
    elif name.lower() == "rosenbrock":
        ModelClass = nessai_models.Rosenbrock
    elif name.lower() in ["gaussianmixture", "gmm"]:
        ModelClass = nessai_models.GaussianMixture
    else:
        raise ValueError(f"Unknown model: {name}")

    return ModelClass(**kwargs)


def run_nessai(
    output: str, model_config: dict, run_kwargs: Optional[None] = None, **kwargs
) -> FlowSampler:

    setup_logger(output=output)

    model = get_model_from_name(**model_config)

    if run_kwargs is None:
        run_kwargs = {}

    fs = FlowSampler(
        model,
        output=output,
        **kwargs,
    )
    fs.run(**run_kwargs)

    return fs
