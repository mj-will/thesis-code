"""Colours for use in my thesis."""
import matplotlib.colors as mc
import numpy as np
import colorsys

teal = "#02979d"
coral_pink = "#f97171"
maroon = "#A10035"
yellow = "#f5b754"

# UofG colours
burgundy = "#7D2239"
cobalt = "#005C8A"
lavender = "#5B4D94"
pillarbox = "#B30C00"
rust = "#9A3A06"
thistle = "#951272"


def lighten_colour(colour, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Based on: https://gist.github.com/ihincks/6a420b599f43fcd7dbd79d56798c4e5a
    """
    try:
        c = mc.cnames[colour]
    except:
        c = colour
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
