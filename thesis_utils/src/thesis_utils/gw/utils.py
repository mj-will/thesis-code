""""Gravitational-wave utilities"""
from typing import List, Union

PRECESSING_SPIN_PARAMETERS = ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl"]
"""List of precessing spin parameters"""


def get_cbc_parameter_labels(
    parameters: Union[List[str], str], units: bool = False, separator: str = "\;"
) -> Union[List[str], str]:
    """Get a list of standard labels for CBC parameters.

    Parameters
    ----------
    parameters :
        List of parameters.

    Returns
    -------
    List of latex labels.
    """
    ref_labels = {
        "mass_ratio": "$q$",
        "chirp_mass": r"$\mathcal{M}$",
        "mass_1": r"$m_1$",
        "mass_2": r"$m_2$",
        "m_1": r"$m_1$",
        "m_2": r"$m_2$",
        "a_1": r"$a_1$",
        "a_2": r"$a_2$",
        "dec": r"$\delta$",
        "ra": r"$\alpha$",
        "zenith": r"$\kappa$",
        "azimuth": r"$\epsilon$",
        "geocent_tme": r"$t_\textrm{c}$",
        "luminosity_distance": r"$d_\textrm{L}$",
        "tilt_1": r"$\theta_1$",
        "tilt_2": r"$\theta_2$",
        "phi_12": r"$\phi_{12}$",
        "phi_jl": r"$\phi_\textrm{JL}$",
        "psi": r"$\psi$",
        "theta_jn": r"$\theta_\textrm{JN}$",
        "phase": r"$\phi_\textrm{c}$",
        "chi_1": r"$\chi_1$",
        "chi_2": r"$\chi_2$",
        "chi_eff": r"$\chi_\textrm{eff}$",
        "chi_p": r"$\chi_{p}$",
        "geocent_time": r"$t_{\textrm{c}}$",
        "H1_time": r"$t_\textrm{H1}$",
        "L1_time": r"$t_\textrm{L1}$",
        "delta_phase": r"$\Delta\phi$",
        "lambda_1": r"$\lambda_1$",
        "lambda_2": r"$\lambda_2$",
        "lambda_tilde": r"$\tilde{\lambda}$",
        "delta_lambda_tilde": r"$\Delta\tilde{\lambda}$",
    }
    ref_units = {
        "chirp_mass": r"[M_{\odot}]",
        "mass_1": r"[M_{\odot}]",
        "mass_2": r"[M_{\odot}]",
        "m_1": r"[M_{\odot}]",
        "m_2": r"[M_{\odot}]",
        "geocent_time": r"[s]",
        "H1_time": r"[s]",
        "L1_time": r"[s]",
        "luminosity_distance": r"[\textrm{Mpc}]",
    }
    if isinstance(parameters, str):
        labels = ref_labels.get(parameters, parameters)
        if units and (u := ref_units.get(parameters, None)):
            labels = labels[:-1] + separator + u + "$"
    else:
        labels = [ref_labels.get(p, p) for p in parameters]
        if units:
            for i, p in enumerate(parameters):
                u = ref_units.get(p, None)
                if u is not None:
                    labels[i] = labels[i][:-1] + separator + u + "$"
    return labels
