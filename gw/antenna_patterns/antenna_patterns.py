"""Definitions of the antenna patterns.

Based on the definitions in: https://arxiv.org/abs/1102.5421
"""
import numpy as np


def f_plus(theta, phi, psi):
    """The F_plus antenna pattern"""
    return 0.5 * (1 + np.cos(theta) ** 2) * np.cos(2 * phi) * np.cos(
        2 * psi
    ) - np.cos(theta) * np.sin(2 * phi) * np.sin(2 * psi)


def f_cross(theta, phi, psi):
    """The F_cross antenna pattern"""
    return 0.5 * (1 + np.cos(theta) ** 2) * np.cos(2 * phi) * np.sin(
        2 * psi
    ) + np.cos(theta) * np.sin(2 * phi) * np.cos(2 * psi)
