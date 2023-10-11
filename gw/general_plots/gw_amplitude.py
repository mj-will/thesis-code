"""Code to compute the amplitude quoted in Section 1.1"""
from astropy import units as u
from astropy.constants import c, G

Qdd = 1000 * u.kilogram * (u.meter**2) / (u.second**2)
dL = (1e6 * u.parsec).to(u.meter)

h = (G / c**4) * (Qdd / dL)
print(f"Amplitude: {h}")
