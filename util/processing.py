import cv2
import numpy as np
from dataclasses import dataclass


# def imread(path):
#     raw = cv2.imread(path)
#     return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


@dataclass
class Box:
    label: str
    x: int
    y: int
    width: int
    height: int

    def to_mask(self, img, b=5):
        h, w = img.shape
        bound_x = min(0, self.x)
        bound_y = min(0, self.y)
        return np.logical_and.outer(
            np.logical_and(np.arange(bound_y, h) >= self.y - b, np.arange(bound_y, h) <= self.y + self.height + 2 * b),
            np.logical_and(np.arange(bound_x, w) >= self.x - b, np.arange(bound_x, w) <= self.x + self.width + 2 * b),
        )


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
