import numpy as np


def bg_to_flake_color(rgbarr):
    """
    Returns the flake color based on an input background color. Values determined empirically.
    :param rgbarr: The RGB array representing the color of the background.
    :return: The RGB array representing the color of the flake.
    """
    red, green, blue = rgbarr
    rval = int(round(0.8643 * red - 2.55, 0))
    gval = int(round(0.8601 * green + 9.6765, 0))
    bval = blue + 4
    # print('coloring')
    return np.array([rval, gval, bval])
