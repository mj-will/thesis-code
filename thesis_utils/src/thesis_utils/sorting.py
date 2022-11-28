"""Sorting utilities"""
import re
from typing import List


def natural_sort(values: List[str]) -> List[str]:
    """Natural sort a list of strings.

    Based on: https://stackoverflow.com/a/4836734
    """
    # fmt: off
    convert = (lambda text: int(text) if text.isdigit() else text.lower())  # noqa: 731
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]  # noqa: 731
    # fmt: on
    return sorted(values, key=alphanum_key)
