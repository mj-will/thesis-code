"""Tests for the sorting utils"""
from thesis_utils.sorting import natural_sort


def test_natural_sort():
    """Assert natural sort returns the correct order"""
    l = ["run_2", "run_1", "run_10", "run_21"]
    expected = ["run_1", "run_2", "run_10", "run_21"]
    assert natural_sort(l) == expected
    assert l != expected
