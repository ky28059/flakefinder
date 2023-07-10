import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from util.logger import logger


def load_config(input_file_path: str) -> list[list[str]]:
    """
    Loads the configuration file at the specified file path.
    :param input_file_path: The path to the config file.
    :return: The parsed list of input and output directories, given in pairs of `[input_dir, output_dir]`
    """
    with open(str(input_file_path)) as file1:
        inputs = file1.readlines()

    config = []
    for line in inputs:
        line = line.strip("\n")
        slicer = line.find("OutputDir:")
        inputdir = line[10:slicer - 2]  # starts after the length of "InputDir: "
        slicer2 = slicer + 11
        outputdir = line[slicer2:]
        config.append([inputdir, outputdir])

    logger.info(f"Loaded config: {config}")
    return config
