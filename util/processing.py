import numpy as np
from dataclasses import dataclass
import cv2
from util.logger import logger


RGB = list[int]
FlakeRGB = np.ndarray[int]


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


# this identifies the edges of flakes, resource-intensive but useful for determining if flake ID is working
def edgefind(imchunk: np.ndarray, avg_rgb: FlakeRGB, pixcals: list[float], t_rgb_dist: int) -> tuple[RGB, any, float]:  # TODO
    """
    TODO
    :param imchunk: The pixels to find edges in.
    :param avg_rgb: The average flake RGB, from `bg_to_flake_color()`.
    :param pixcals:
    :param t_rgb_dist: The threshold a pixel color must be within from the average flake color to be counted as good.
    :return: The results, as a tuple of (flake rgb, edge image, flake area).
    """
    pixcalw, pixcalh = pixcals
    edgerad = 20

    imchunk2 = imchunk.copy()
    impix = imchunk.copy().reshape(-1, 3)
    dims = np.shape(imchunk)

    flakeid = np.sqrt(np.sum((impix - avg_rgb) ** 2, axis=1)) < t_rgb_dist  # a mask for pixel color
    maskpic = np.reshape(flakeid, (dims[0], dims[1], 1))

    red_freq = np.bincount(impix[:, 0] * flakeid)
    green_freq = np.bincount(impix[:, 1] * flakeid)
    blue_freq = np.bincount(impix[:, 2] * flakeid)
    red_freq[0] = 0  # otherwise argmax finds values masked to 0 by flakeid
    green_freq[0] = 0
    blue_freq[0] = 0

    # determines flake RGB as the most common R,G,B value in identified flake region
    rgb = [red_freq.argmax(), green_freq.argmax(), blue_freq.argmax()]

    h, w, c = imchunk.shape
    flakeid2 = np.sqrt(np.sum((impix - rgb) ** 2, axis=1)) < 5  # a mask for pixel color
    maskpic2 = np.reshape(flakeid2, (dims[0], dims[1], 1))
    indices = np.argwhere(np.any(maskpic2 > 0, axis=2))  # flake region
    farea = round(len(indices) * pixcalw * pixcalh, 1)

    grayimg = cv2.cvtColor(imchunk, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.fastNlMeansDenoising(grayimg, None, 2, 3, 11)
    edgeim = np.reshape(cv2.Canny(grayimg, 5, 15), (h, w, 1)) \
               .astype(np.int16) * np.array([25, 25, 25]) / 255

    return rgb, edgeim.astype(np.uint8), farea


def merge_boxes(dbscan_img, boxes: list[Box], eliminated_indexes: list[int] = []) -> list[Box]:
    """
    Merges a list of boxes by combining boxes with overlap.
    :param dbscan_img: TODO
    :param boxes: The list of boxes to merge.
    :param eliminated_indexes: TODO
    :return: The merged list.
    """
    merged = []

    for _i in range(len(boxes)):
        if _i in eliminated_indexes:
            continue
        i = boxes[_i]
        for _j in range(_i + 1, len(boxes)):
            j = boxes[_j]
            # Ith box is always <= jth box regarding y. Not necessarily w.r.t x.
            # sequence the y layers.
            # just cheat and use Intersection in pixel space method.
            on_i = i.to_mask(dbscan_img)
            on_j = j.to_mask(dbscan_img)

            # Now calculate their intersection. If there's any overlap we'll count that.
            intersection_count = np.logical_and(on_i, on_j).sum()

            if intersection_count > 0:
                # Extend the first box to include dimensions of the 2nd box.
                x_min = min(i.x, j.x)
                x_max = max(i.x + i.width, j.x + j.width)
                y_min = min(i.y, j.y)
                y_max = max(i.y + i.height, j.y + j.height)

                new_width = x_max - x_min
                new_height = y_max - y_min
                i = Box(i.label, x_min, y_min, new_width, new_height)
                eliminated_indexes.append(_j)
        merged.append(i)

    return merged
