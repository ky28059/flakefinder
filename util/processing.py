import cv2
import numpy as np

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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
    morph_size = 7

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * morph_size + 1, 2 * morph_size + 1))
    return cv2.morphologyEx(masked, cv2.MORPH_OPEN, element)


def apply_morph_close(masked):
    """
    Applies the "closing" morphological operation to a masked image to fill eroded flake chunks from opening.
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    :param masked: The masked black and white image from `mask_flake_color`.
    :return: The image, with the morph applied.
    """
    morph_size = 14

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * morph_size + 1, 2 * morph_size + 1))
    return cv2.morphologyEx(masked, cv2.MORPH_CLOSE, element)
