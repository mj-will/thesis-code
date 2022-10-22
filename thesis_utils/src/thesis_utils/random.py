"""Utilities related to randomness"""
import random
from typing import Optional

import numpy as np
import torch

from . import conf


def seed_everything(seed: Optional[int] = None) -> None:
    """Seed all the random number generation.

    Seeds `random`, `numpy` and `torch` and records the seed in `conf.seed`.
    """
    if seed is None:
        seed = conf.default_seed
    conf.seed = seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
