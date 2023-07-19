import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np

from config import k
from util.box import make_boxes, merge_boxes, draw_box
from util.processing import bg_to_flake_color, get_avg_rgb, mask_flake_color, apply_morph_open, apply_morph_close, get_angles


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

    for s in args.s:
        # Run all the flake color logic first, since that isn't what's being benchmarked here
        # TODO: don't hard-code the input directory?
        img = cv2.imread(f"C:\\04_03_23_EC_1\\Scan 002\\TileScan_001\\TileScan_001--Stage{s}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lowlim = np.array([87, 100, 99])  # defines lower limit for what code can see as background
        highlim = np.array([114, 118, 114])

        imsmall = cv2.resize(img.copy(), dsize=(256 * k, 171 * k)).reshape(-1, 3)
        test = np.sign(imsmall - lowlim) + np.sign(highlim - imsmall)
        pixout = imsmall * np.sign(test + abs(test))

        back_rgb = get_avg_rgb(pixout)
        flake_avg_rgb = bg_to_flake_color(back_rgb)
        flake_avg_hsv = cv2.cvtColor(np.uint8([[flake_avg_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?

        # Benchmark cv2 thresholding
        tik = time.time()
        masked = mask_flake_color(img, flake_avg_hsv)
        tok = time.time()

        name = str(time.time_ns())
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, masked)
        cv2.waitKey()

        print(f"Finished threshold mask for {s} in {tok - tik} seconds")

        tik = time.time()
        dst = apply_morph_close(masked)
        tok = time.time()

        cv2.imshow(name, dst)
        cv2.waitKey()

        print(f"Finished close morph in {tok - tik} seconds")

        tik = time.time()
        dst = apply_morph_open(dst)
        tok = time.time()

        cv2.imshow(name, dst)
        cv2.waitKey()

        print(f"Finished open morph in {tok - tik} seconds")

        tik = time.time()
        contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        dst = cv2.drawContours(cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), contours, -1, (0, 0, 255), 2)
        tok = time.time()

        cv2.imshow(name, dst)
        cv2.waitKey()

        print(f"Finished contour search in {tok - tik} seconds")

        tik = time.time()
        boxes = make_boxes(contours, img.shape[0], img.shape[1])
        boxes = merge_boxes(masked, boxes)
        # boxes = merge_boxes(masked, boxes)
        tok = time.time()

        print(f"Generated and merged boxes in {tok - tik} seconds")

        for box in boxes:
            img = draw_box(img, box)
            img = cv2.drawContours(img, box.contours, -1, (255, 255, 255), 1)

            mask = np.zeros(img.shape, np.uint8)
            mask = cv2.drawContours(mask, box.contours, -1, (255, 255, 255), 1)

            # TODO: make the mask b&w to begin with
            lines = cv2.HoughLinesP(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 1, np.pi / 180, 50, None, 50, 10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img, (x1, y1), (x2, y2), (192, 8, 254), 2, cv2.LINE_AA)

                if len(lines) < 2:
                    continue

                angles = get_angles(lines)
                for i in range(len(angles)):
                    cv2.putText(img, str(round(np.rad2deg(angles[i]), 2)) + ' deg.',
                                (box.x + box.width + 10, box.y + int(box.height / 2) + (i + 1) * 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

        # baseline = cv2.imread(f"C:\\04_03_23_EC_1\\MLScanned2\\TileScan_001--Stage{s}.jpg")
        # cv2.imshow(name, baseline)
        # cv2.waitKey()

        print("-----")
