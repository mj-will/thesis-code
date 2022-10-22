"""Tests for the rejection module"""
from nessai_models import Gaussian
import pytest

from thesis_utils.sampling.rejection import rejection_sample_model


@pytest.mark.parametrize("n", [10, 100])
def test_rejection_sample_model(n):
    model = Gaussian()
    samples = rejection_sample_model(model, n)
    assert len(samples) == n
