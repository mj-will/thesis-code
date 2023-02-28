"""Class for lazily loading injections"""
import dataclasses
from typing import Optional

from ...io import load_json


@dataclasses.dataclass
class Injection:
    """Dataclass to store injection parameters"""

    injection_file: str
    """The injection file"""

    chirp_mass: Optional[float] = None
    mass_ratio: Optional[float] = None
    total_mass: Optional[float] = None
    mass_1: Optional[float] = None
    mass_2: Optional[float] = None
    a_1: Optional[float] = None
    a_2: Optional[float] = None
    tilt_1: Optional[float] = None
    tilt_2: Optional[float] = None
    phi_12: Optional[float] = None
    phi_jl: Optional[float] = None
    theta_jn: Optional[float] = None
    phase: Optional[float] = None
    psi: Optional[float] = None
    ra: Optional[float] = None
    dec: Optional[float] = None
    luminosity_distance: Optional[float] = None
    geocent_time: Optional[float] = None

    def __post_init__(self):
        data = load_json(self.injection_file)
        current = dataclasses.asdict(self)
        for key, value in data.items():
            if not key in current:
                raise KeyError(f"Invalid key: {key}")
            if current[key] is None:
                setattr(self, key, value)

    def bilby_format(self):
        """Return a dictionary with the correct format for bilby"""
        return {
            key: value
            for key, value in dataclasses.asdict(self).items()
            if value is not None
        }
