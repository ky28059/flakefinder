import numpy as np
from dataclasses import dataclass
from util.logger import logger


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


# this identifies the edges of flakes, resource-intensive but useful for determining if flake ID is working
def edgefind(imchunk, avg_rgb, pixcals: list[float], t_rgb_dist: int) -> tuple[list[int], list[int], float]:
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

    flakeid2 = np.sqrt(np.sum((impix - rgb) ** 2, axis=1)) < 5  # a mask for pixel color
    maskpic2 = np.reshape(flakeid2, (dims[0], dims[1], 1))

    indices = np.argwhere(np.any(maskpic2 > 0, axis=2))  # flake region
    farea = round(len(indices) * pixcalw * pixcalh, 1)

    # TODO: rename
    indices3 = [
        index for index in np.argwhere(np.any(maskpic2 > -1, axis=2))
        if 3 < np.min(np.sum((indices - index) ** 2, axis=1)) < 20
    ]

    logger.info('boundary found')
    return rgb, indices3, farea


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
                # print(x_min, x_max)
                # print(y_min, y_max)
                new_width = x_max - x_min
                new_height = y_max - y_min
                i = Box(i.label, x_min, y_min, new_width, new_height)
                eliminated_indexes.append(_j)
        merged.append(i)

    return merged
