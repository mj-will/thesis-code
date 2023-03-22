""""Gravitational-wave utilities"""
from typing import List, Union


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
        "mass_1": r"m_1",
        "mass_2": r"m_2",
        "a_1": r"$\chi_1$",
        "a_2": r"$\chi_2$",
        "dec": r"$\delta$",
        "ra": r"$\alpha$",
        "geocent_tme": r"$t_\textrm{c}$",
        "luminosity_distance": r"$d_\textrm{L}$",
        "tilt_1": r"$\theta_1$",
        "tilt_2": r"$\theta_2$",
        "phi_12": r"$\phi_{12}$",
        "phi_jl": r"$\phi_{JL}$",
        "psi": r"$\psi$",
        "theta_jn": r"$\theta_{JN}$",
        "phase": r"$\phi$",
        "chi_1": r"$\chi_1$",
        "chi_2": r"$\chi_2$",
        "geocent_time": r"$t_{\textrm{c}}$",
        "delta_phase": r"$\Delta\phi$",
    }
    units = {
        "chirp_mass": r"[M_{\odot}]",
        "geocent_time": r"[s]",
        "luminosity_distance": r"[\textrm{Mpc}]",
    }
    if isinstance(parameters, str):
        labels = ref_labels.get(parameters, parameters)
        if units and (u := units.get(parameters, None)):
            labels = labels[:-1] + separator + u + "$"
    else:
        labels = [ref_labels.get(p, p) for p in parameters]
        if units:
            for i, p in enumerate(parameters):
                u = units.get(p, None)
                if u is not None:
                    labels[i] = labels[i][:-1] + separator + u + "$"
    return labels
