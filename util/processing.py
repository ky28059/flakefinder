import cv2
import numpy as np

from config import open_morph_size, close_morph_size, open_morph_shape, close_morph_shape, flake_angle_tolerance_rads, k

RGB = list[int]
FlakeRGB = np.ndarray[int]


def bg_to_flake_color(rgb: RGB) -> FlakeRGB:
    """
    Returns the flake color based on an input background color. Values determined empirically.
    :param rgb: The RGB array representing the color of the background.
    :return: The RGB array representing the color of the flake.
    """
    red, green, blue = rgb

    flake_red = int(round(0.8643 * red - 2.55, 0))
    flake_green = int(round(0.8601 * green + 9.6765, 0))
    flake_blue = blue + 4

    return np.array([flake_red, flake_green, flake_blue])


def get_bg_pixels(img: np.ndarray):
    # Lower and higher RGB limits for what code can see as background
    lowlim = np.array([87, 100, 99])
    highlim = np.array([114, 118, 114])

    imsmall = cv2.resize(img.copy(), dsize=(256 * k, 171 * k)).reshape(-1, 3)
    test = np.sign(imsmall - lowlim) + np.sign(highlim - imsmall)
    return imsmall * np.sign(test + abs(test))


def get_avg_rgb(img: np.ndarray, mask: np.ndarray[bool] = 1) -> RGB:
    """
    Gets the average RGB within a given array of RGB values.
    :param img: The image to process.
    :param mask: An optional mask to apply to RGB values.
    :return: The average RGB.
    """
    red_freq = np.bincount(img[:, 0] * mask)
    green_freq = np.bincount(img[:, 1] * mask)
    blue_freq = np.bincount(img[:, 2] * mask)

    red_freq[0] = 0  # otherwise argmax finds values masked to 0
    green_freq[0] = 0
    blue_freq[0] = 0

    return [red_freq.argmax(), green_freq.argmax(), blue_freq.argmax()]


def mask_flake_color(img: np.ndarray, flake_avg_hsv: np.ndarray) -> np.ndarray:
    """
    Mask an image to black and white pixels based on whether it is within threshold of the given flake color.
    :param flake_avg_hsv: The average flake color (in HSV).
    :param img: The image to mask.
    :return: The masked black and white image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = tuple(map(int, flake_avg_hsv - (6, 25, 25)))
    higher = tuple(map(int, flake_avg_hsv + (6, 25, 25)))

    return cv2.inRange(hsv, lower, higher)


def apply_morph_open(masked):
    """
    Applies the "opening" morphological operation to a masked image to clear away small false-positive "islands".
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    :param masked: The masked black and white image from `mask_flake_color`.
    :return: The image, with the morph applied.
    """
    element = cv2.getStructuringElement(open_morph_shape, (2 * open_morph_size + 1, 2 * open_morph_size + 1))
    return cv2.morphologyEx(masked, cv2.MORPH_OPEN, element)


def apply_morph_close(masked):
    """
    Applies the "closing" morphological operation to a masked image to fill eroded flake chunks from opening.
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    :param masked: The masked black and white image from `mask_flake_color`.
    :return: The image, with the morph applied.
    """
    element = cv2.getStructuringElement(close_morph_shape, (2 * close_morph_size + 1, 2 * close_morph_size + 1))
    return cv2.morphologyEx(masked, cv2.MORPH_CLOSE, element)


def in_bounds(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> bool:
    """
    Gets if a flake bounded by (x1, y1) and (x2, y2) is entirely contained in another image.

    :param x1: The lower-left x coordinate.
    :param y1: The lower-left y coordinate.
    :param x2: The upper-right x coordinate.
    :param y2: The upper-right y coordinate.
    :param w: The width of the box.
    :param h: The height of the box.
    :return: Whether the flake is entirely contained in another image.
    """
    delt = 0.05
    return x2 > delt * w and y2 > delt * h and x1 < (1 - delt) * w and y1 < (1 - delt) * h


def get_angles(lines: np.ndarray[tuple[tuple[float, float, float, float]]]) -> list[float]:
    """
    Gets all angles within a given range of a multiple of 30 degrees (excluding 180 and 360) given a list of lines.
    :param lines: The lines to get angles from.
    :return: The list of filtered angles (in radians).
    """
    ret = []

    for i in range(0, len(lines)):
        for j in range(i, len(lines)):
            x11, y11, x21, y21 = lines[i][0]
            x12, y12, x22, y22 = lines[j][0]

            # Calculate angle between lines
            t1 = np.arctan2(x21 - x11, y21 - y11)
            t2 = np.arctan2(x22 - x12, y22 - y12)
            t = (t2 - t1) % (2 * np.pi)

            if t % (np.pi / 6) > flake_angle_tolerance_rads or t < flake_angle_tolerance_rads or t > 2 * np.pi - flake_angle_tolerance_rads:
                continue

            ret.append(t)

    return ret
