"""Functions for rejection sampling"""

from nessai.livepoint import empty_structured_array
from nessai.model import Model
import numpy as np

from .. import conf


def rejection_sample_model(model: Model, n: int) -> np.ndarray:
    """Use rejection sampling to sample from a model"""
    n_sampled = 0
    samples = empty_structured_array(n, model.names)
    rng = np.random.default_rng(seed=conf.seed)
    while n_sampled < n:
        x = model.new_point(n)
        log_proposal = model.new_point_log_prob(x)
        log_target = model.log_likelihood(x)
        log_w = log_target - log_proposal
        log_w -= log_w.max()
        log_u = np.log(rng.random(log_w.size))
        acc = log_w > log_u
        m = min(np.sum(acc), n - n_sampled)
        samples[n_sampled : (n_sampled + m)] = x[acc][:m]
        n_sampled += m
    return samples
