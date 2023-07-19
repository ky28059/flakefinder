from dataclasses import dataclass

import cv2
import numpy as np

from config import t_min_cluster_pixel_count
from util.processing import in_bounds


@dataclass
class Box:
    contours: np.ndarray
    area: int
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


def make_boxes(contours, hierarchy, img_h: int, img_w: int) -> list[Box]:
    """
    Make boxes from contours, filtering out contours that are too small or completely contained by another image.
    :param img_h: The height of the image.
    :param img_w: The width of the image.
    :param contours: The contours to draw boxes from.
    :return: The list of boxes.
    """
    boxes = []
    inner_indices = []

    for i in range(len(contours)):
        if i in inner_indices:
            continue

        cnt = contours[i]
        _, _, child, parent = hierarchy[0][i]

        area = cv2.contourArea(cnt)

        # Subtract child contours to better represent area
        # TODO: recursion to children of children?
        # TODO: does the "count all children on same level" work with multiple nested contours?
        while child != -1:
            child_cnt = contours[child]

            area -= cv2.contourArea(child_cnt)

            cnt = np.concatenate([cnt, child_cnt])  # Add the child contour to `cnt` so it shows up in the final image
            inner_indices.append(child)

            # Loop to next contour on same level as defined by hierarchy tree, if it exists
            child, _, _, _ = hierarchy[0][child]

        if area < t_min_cluster_pixel_count:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if not in_bounds(x, y, x + w, y + h, img_w, img_h):
            continue

        boxes.append(Box(cnt, area, x, y, w, h))

    return boxes


def merge_boxes(dbscan_img, boxes: list[Box]) -> list[Box]:
    """
    Merges a list of boxes by combining boxes with overlap.
    :param dbscan_img: TODO
    :param boxes: The list of boxes to merge.
    :return: The merged list.
    """
    merged = []
    eliminated_indexes = []

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
                i = Box(
                    np.concatenate([i.contours, j.contours]),
                    i.area + j.area,
                    x_min, y_min,
                    new_width, new_height
                )
                eliminated_indexes.append(_j)
        merged.append(i)

    return merged


offset = 5
color = (255, 0, 0)
thickness = 6
font = cv2.FONT_HERSHEY_SIMPLEX


def draw_box(img: np.ndarray, b: Box) -> np.ndarray:
    pixcal = 1314.08 / img.shape[1]  # microns/pixel from Leica calibration

    offset_x = int(b.x) - offset
    offset_y = int(b.y) - offset
    offset_w = int(b.width) + 2 * offset
    offset_h = int(b.height) + 2 * offset

    width_microns = round(offset_w * pixcal, 1)
    height_microns = round(offset_h * pixcal, 1)  # microns

    img = cv2.rectangle(img, (offset_x, offset_y), (offset_x + offset_w, offset_y + offset_h), color, thickness)
    img = cv2.putText(img, str(height_microns), (offset_x + offset_w + 10, offset_y + int(offset_h / 2)), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    img = cv2.putText(img, str(width_microns), (offset_x, offset_y - 10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return img
