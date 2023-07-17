import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np

from config import t_min_cluster_pixel_count, k
from util.processing import bg_to_flake_color, get_avg_rgb, mask_flake_color, apply_morph_open, apply_morph_close


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
        img0 = cv2.imread(f"C:\\04_03_23_EC_1\\Scan 002\\TileScan_001\\TileScan_001--Stage{s}.jpg")
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

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
        masked = mask_flake_color(img0, flake_avg_hsv)
        tok = time.time()

        cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
        cv2.imshow("threshold", masked)
        cv2.waitKey()

        print(f"Finished threshold mask for {s} in {tok - tik} seconds")

        tik = time.time()
        dst = apply_morph_open(masked)
        tok = time.time()

        cv2.imshow("threshold", dst)
        cv2.waitKey()

        print(f"Finished open morph in {tok - tik} seconds")

        tik = time.time()
        dst = apply_morph_close(dst)
        tok = time.time()

        cv2.imshow("threshold", dst)
        cv2.waitKey()

        print(f"Finished close morph in {tok - tik} seconds")

        tik = time.time()
        contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        dst = cv2.drawContours(cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), contours, -1, (0, 0, 255), 2)
        tok = time.time()

        cv2.imshow("threshold", dst)
        cv2.waitKey()

        mask = np.zeros(img.shape, np.uint8)
        mask = cv2.drawContours(mask, contours, -1, (255, 255, 255), 1)

        cv2.imshow("threshold", mask)
        cv2.waitKey()

        # TODO: make the mask b&w to begin with
        lines = cv2.HoughLinesP(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 1, np.pi / 180, 50, None, 50, 10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (192, 8, 254), 2, cv2.LINE_AA)

        cv2.imshow("threshold", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

        for cnt in contours:
            if cv2.contourArea(cnt) < t_min_cluster_pixel_count:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if lines is not None:
                # Filter (and unpack) lines contained fully within the bounding box
                filtered = [
                    line[0] for line in lines
                    if line[0, 0] >= x and line[0, 1] >= y and line[0, 2] <= x + w and line[0, 3] <= y + h
                ]
                if len(filtered) < 2:
                    continue

                # TODO
                x11, y11, x21, y21 = filtered[0]
                x12, y12, x22, y22 = filtered[1]

                t1 = np.arctan2(x21 - x11, y21 - y11)
                t2 = np.arctan2(x22 - x12, y22 - y12)
                t = (t2 - t1) % (2 * np.pi)

                img3 = cv2.putText(img, str(round(np.rad2deg(t), 2)) + '°', (x + w + 10, y + int(h / 2)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("threshold", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

        # baseline = cv2.imread(f"C:\\04_03_23_EC_1\\MLScanned2\\TileScan_001--Stage{s}.jpg")
        # cv2.imshow("threshold", baseline)
        # cv2.waitKey()

        print(f"Finished contour search in {tok - tik} seconds")
        print("-----")
