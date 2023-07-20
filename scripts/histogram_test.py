import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np

from config import k
from util.queue import load_queue
from util.processing import bg_to_flake_color, get_avg_rgb, mask_flake_color

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')



def make_histogram(x, title: str, n: int, back: int, flake: int, lim: int = 256):
    print(f"Plotting {title}")

    if n != -1:
        plt.subplot(3, 1, n)
    plt.plot(x)
    plt.axvline(back, dashes=(2, 1))
    plt.axvline(flake, dashes=(2, 1), color='b')

    plt.ylabel('Occurrences')
    plt.xlim([0, lim])
    plt.title(title)


def show_img(img, mask=None):
    reds = cv2.calcHist([img], [0], mask, [256], [0, 256])
    greens = cv2.calcHist([img], [1], mask, [256], [0, 256])
    blues = cv2.calcHist([img], [2], mask, [256], [0, 256])

    masked = " (masked)" if mask is not None else ""

    make_histogram(reds, f"Stage {s} -- reds" + masked, 1, back_rgb[0], flake_avg_rgb[0])
    make_histogram(greens, f"Stage {s} -- greens" + masked, 2, back_rgb[1], flake_avg_rgb[1])
    make_histogram(blues, f"Stage {s} -- blues" + masked, 3, back_rgb[2], flake_avg_rgb[2])
    plt.show()

    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hues = cv2.calcHist([img2], [0], mask, [180], [0, 180])
    saturations = cv2.calcHist([img2], [1], mask, [256], [0, 256])
    values = cv2.calcHist([img2], [2], mask, [256], [0, 256])

    make_histogram(hues, f"Stage {s} -- hues" + masked, 1, back_hsv[0], flake_avg_hsv[0], lim=180)
    make_histogram(saturations, f"Stage {s} -- saturations" + masked, 2, back_hsv[1], flake_avg_hsv[1])
    make_histogram(values, f"Stage {s} -- values" + masked, 3, back_hsv[2], flake_avg_hsv[2])
    plt.show()

    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    intensities = cv2.calcHist([img3], [0], mask, [256], [0, 256])

    make_histogram(intensities, f"Stage {s} -- intensities" + masked, -1, back_gray, flake_avg_gray)
    plt.show()


if __name__ == "__main__":
    # TODO: new description or abstract
    parser = argparse.ArgumentParser(
        description="Find graphene flakes on SiO2. Currently configured only for exfoliator dataset"
    )
    parser.add_argument(
        "--s",
        required=True,
        type=int,
        nargs="+",
        help="Scan stages to test (ex. --s 246 248 250)"
    )
    args = parser.parse_args()

    queue = load_queue('Queue.txt')
    input_dir, _ = queue[0]

    for s in args.s:
        img = cv2.imread(f"{input_dir}\\TileScan_001--Stage{s}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Calculate background and flake RGB using old method for labelling purposes
        lowlim = np.array([87, 100, 99])  # defines lower limit for what code can see as background
        highlim = np.array([114, 118, 114])

        imsmall = cv2.resize(img.copy(), dsize=(256 * k, 171 * k)).reshape(-1, 3)
        test = np.sign(imsmall - lowlim) + np.sign(highlim - imsmall)
        pixout = imsmall * np.sign(test + abs(test))

        back_rgb = get_avg_rgb(pixout)
        back_hsv = cv2.cvtColor(np.uint8([[back_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?
        back_gray = cv2.cvtColor(np.uint8([[back_rgb]]), cv2.COLOR_RGB2GRAY)[0][0]  # TODO: hacky?

        flake_avg_rgb = bg_to_flake_color(back_rgb)
        flake_avg_hsv = cv2.cvtColor(np.uint8([[flake_avg_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?
        flake_avg_gray = cv2.cvtColor(np.uint8([[flake_avg_rgb]]), cv2.COLOR_RGB2GRAY)[0][0]  # TODO: hacky?

        # Calculate histograms from image
        show_img(img)

        mask = mask_flake_color(img, flake_avg_hsv)
        show_img(img, mask)
