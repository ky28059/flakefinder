from dataclasses import dataclass
import numpy as np


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
