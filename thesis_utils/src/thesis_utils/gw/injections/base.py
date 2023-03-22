"""Class for lazily loading injections"""
import dataclasses
from typing import Optional

from ...io import load_json
from ..utils import PRECESSING_SPIN_PARAMETERS


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
    chi_1: Optional[float] = None
    chi_2: Optional[float] = None
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

    def convert_to_aligned(self):
        """Convert to aligned spin.

        Returns a copy with the precessing spin parameters set to None.
        """
        replace = {k: None for k in PRECESSING_SPIN_PARAMETERS}
        out = dataclasses.replace(self, **replace)

        for i in [1, 2]:
            c = f"chi_{i}"
            a = f"a_{i}"
            if getattr(self, c) is None:
                if (val := getattr(self, a)) is not None:
                    setattr(out, c, val)
                else:
                    raise ValueError("Missing spin magnitude")
        return out

    def bilby_format(self):
        """Return a dictionary with the correct format for bilby"""
        d = {
            key: value
            for key, value in dataclasses.asdict(self).items()
            if value is not None
        }
        d.pop("injection_file")
        return d
