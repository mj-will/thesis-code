"""Custom functions for sampling with the quaternions"""
from typing import Dict, List, Union

from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    _generate_all_cbc_parameters,
)
from bilby.gw.likelihood import GravitationalWaveTransient
import numpy as np


class GravitationalWaveTransientWithQuaternions(GravitationalWaveTransient):
    """Wrapper for standard GW class in bilby that includes quaternions."""

    def log_likelihood_ratio(self):
        # Add psi for the antenna patterns
        if "q_0" in self.parameters:
            self.parameters["psi"] = quaternion_to_psi(
                np.array(
                    [
                        self.parameters["q_0"],
                        self.parameters["q_1"],
                        self.parameters["q_2"],
                        self.parameters["q_3"],
                    ]
                )
            )
        return super().log_likelihood_ratio()


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to Euler angles. Quaternion does not need to
    be normalised.

    See the quaternion package for the original implementation:
    https://github.com/moble/quaternion

    Parameters
    ----------
    q: array_like
        Quaternion to convert to Euler angles

    Returns
    -------
    alpha_beta_gamma: array_like
        Array of Euler angles
    """
    n = np.sum(q**2, axis=-1)
    alpha_beta_gamma = np.empty((q.shape[0], 3), dtype=np.float)
    alpha_beta_gamma[..., 0] = np.arctan2(q[..., 3], q[..., 0]) + np.arctan2(
        -q[..., 1], q[..., 2]
    )
    alpha_beta_gamma[..., 1] = 2 * np.arccos(
        np.sqrt((q[..., 0] ** 2 + q[..., 3] ** 2) / n)
    )
    alpha_beta_gamma[..., 2] = np.arctan2(q[..., 3], q[..., 0]) - np.arctan2(
        -q[..., 1], q[..., 2]
    )
    return alpha_beta_gamma


def quaternion_to_psi(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to polarisation angle psi

    Parameters
    ----------
    quat: array_like
        Quaternion to convert to source angles

    Returns
    -------
    psi: array_like
        Array of psi values
    """
    if q.ndim == 1:
        q = q[np.newaxis, :]
    if q.shape[1] != 4:
        raise RuntimeError("A quaternion must have four components")
    psi = np.empty(q.shape[0], dtype=np.float)
    # psi[...] = np.arctan2(q[..., 3], q[..., 0]) - np.arctan2(- q[..., 1], q[..., 2])   # for gamma
    psi[...] = np.arctan2(q[..., 3], q[..., 0]) + np.arctan2(
        -q[..., 1], q[..., 2]
    )  # for alpha
    psi[...] = psi[...] % (2.0 * np.pi) % np.pi
    return np.squeeze(psi)


def quaternion_to_source_angles(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to source angles: polarisation, inclination,
    phase. Quaternion does not need to be normalised.

    Parameters
    ----------
    quat: array_like
        Quaternion to convert to source angles

    Returns
    -------
    angles: array_like
        Array of angles: polarisation, inclination, phase
    """
    if q.ndim == 1:
        q = q[np.newaxis, :]
    if q.shape[1] != 4:
        raise RuntimeError("A quaternion must have four components")
    angles = quaternion_to_euler(q)
    angles[...] = angles[...] % (2.0 * np.pi)
    # for psi
    angles[..., 0] = angles[..., 0] % np.pi
    return np.squeeze(angles)


def convert_with_quaternions(parameters: Dict) -> Union[Dict, List[str]]:
    """Wrapper for the bilby conversion function that adds quaternions"""
    original_keys = list(parameters.keys())
    converted_parameters, _ = convert_to_lal_binary_black_hole_parameters(parameters)

    if "q_0" in converted_parameters.keys():
        quat = np.array(
            [
                converted_parameters["q_0"],
                converted_parameters["q_1"],
                converted_parameters["q_2"],
                converted_parameters["q_3"],
            ]
        )

        angles = quaternion_to_source_angles(quat.T)
        converted_parameters["psi"] = angles[..., 0]
        converted_parameters["theta_jn"] = angles[..., 1]
        converted_parameters["phase"] = angles[..., 2]

    added_keys = [
        key for key in converted_parameters.keys() if key not in original_keys
    ]

    return converted_parameters, added_keys


def generate_all_bbh_parameters(sample, likelihood=None, priors=None, npool=1):
    """
    From either a single sample or a set of samples fill in all missing
    BBH parameters, in place.

    Parameters
    ----------
    sample: dict or pandas.DataFrame
        Samples to fill in with extra parameters, this may be either an
        injection or posterior samples.
    likelihood: bilby.gw.likelihood.GravitationalWaveTransient, optional
        GravitationalWaveTransient used for sampling, used for waveform and
        likelihood.interferometers.
    priors: dict, optional
        Dictionary of prior objects, used to fill in non-sampled parameters.
    """
    waveform_defaults = {
        "reference_frequency": 50.0,
        "waveform_approximant": "IMRPhenomPv2",
        "minimum_frequency": 20.0,
    }
    output_sample = _generate_all_cbc_parameters(
        sample,
        defaults=waveform_defaults,
        base_conversion=convert_with_quaternions,
        likelihood=likelihood,
        priors=priors,
        npool=npool,
    )
    return output_sample
