"""
Utilities for coordinate transforms.
"""
import numpy as np


def spherical_to_cartesian(theta, phi, radius):
    """Convert spherical polar coordinates to Cartesian.

    The convention is theta ~ [0, pi], phi ~ [0, 2pi].
    """
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z
