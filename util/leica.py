import numpy as np
import re
import argparse


Dimensions = tuple[int, int]


def pos_get(input_dir: str) -> np.ndarray[float]:
    """
    Gets the scan dimensions and positions contained in a leica metadata file.
    :param input_dir: The directory containing the microscope file.
    :return: A 2d numpy array representing a list of [x dimension, x position (mm), y dimension, y position (mm)]
    """
    with open(input_dir + "/leicametadata/TileScan_001.xlif", 'r') as file:
        rawdata = file.read()

    # Match metadata <Tile> tags in the form of <Tile FieldX="..." FieldY="..." PosX="..." PosY="..." PosZ="..." />
    matches = re.findall(r"<Tile FieldX=\"(.+)\" FieldY=\"(.+)\" PosX=\"(.+)\" PosY=\"(.+)\" PosZ=\".+\" />", rawdata)
    return np.array([[int(xd), float(posx), int(yd), float(posy)] for xd, yd, posx, posy in matches])


def dim_get(input_dir: str) -> Dimensions:
    """
    Gets the scan dimensions from a microscope file.
    :param input_dir: The directory containing the microscope file.
    :return: The dimensions of the scan, as a tuple of [width, height].
    """
    data = pos_get(input_dir)

    xmax = int(np.max(data[:, 0]))
    ymax = int(np.max(data[:, 2]))

    return xmax + 1, ymax + 1


def location(m: int, dims: Dimensions) -> tuple[float, int, int, int]:  # TODO: supposed to be a float?
    """
    Gets the (x, y) location of the mth scan image given the width and height of the scan.
    :param m: The index to locate.
    :param dims: The dimensions of the scan, as a tuple of (width, height).
    :return: The location, as a tuple of (x, y, height - 1, width - 1)
    """
    width, height = dims
    row = m % height
    column = (m - row) / height
    print(f"Location for {m}: {column}, {row}")
    return column, row, height - 1, width - 1


if __name__ == '__main__':
    from logger import logger
    from config import load_config

    # TODO: new description or abstract
    parser = argparse.ArgumentParser(
        description="Find graphene flakes on SiO2. Currently configured only for exfoliator dataset"
    )
    parser.add_argument(
        "--q",
        required=True,
        type=str,
        help="Directory containing images to process. Optional unless running in headless mode"
    )
    args = parser.parse_args()

    for input_dir, _ in load_config(args.q):
        logger.info(f"Found dimensions for {input_dir}: {dim_get(input_dir)}")
