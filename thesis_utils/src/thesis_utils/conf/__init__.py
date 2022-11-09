"""Global configuration."""
import numpy as np

default_seed = 427
"""The default random seed"""
seed = None
"""The current random seed"""

figure_ftype = "pdf"
"""File type for saving figures."""
base_colour = "#02979d"
"""Default base colour"""
highlight_colour = "#f5b754"
"""Default colour for highlights"""
default_corner_kwargs = dict(
    bins=32,
    smooth=0.9,
    color=base_colour,
    truth_color=highlight_colour,
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)),
    plot_density=True,
    plot_datapoints=True,
    fill_contours=True,
    show_titles=True,
    hist_kwargs=dict(density=True),
)
"""Default corner plot keyword arguments"""
