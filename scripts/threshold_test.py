import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np

from util.processing import bg_to_flake_color, get_avg_rgb, mask_flake_color, apply_morph_open, apply_morph_close

k = 4
t_rgb_dist = 8
t_min_cluster_pixel_count = 30 * 25  # flake too small


def classical(img0):
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    img_pixels = img.copy().reshape(-1, 3)
    rgb_pixel_dists = np.sqrt(np.sum((img_pixels - flake_avg_rgb) ** 2, axis=1))

    img_mask = np.logical_and(rgb_pixel_dists < t_rgb_dist, back_rgb[0] - img_pixels[:, 0] > 5)
    # t_count = np.sum(img_mask)

    img2_mask_in = img.copy().reshape(-1, 3)
    img2_mask_in[~img_mask] = np.array([0, 0, 0])

    return img2_mask_in.reshape(img.shape)


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

        # Benchmark classical
        tik = time.time()
        masked = classical(img0)
        tok = time.time()

        cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
        cv2.imshow("threshold", cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

        print(f"Finished classical mask for {s} in {tok - tik} seconds")
        print("-----")

        # tik = time.time()
        # dbscan_img = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
        # dbscan_img = cv2.resize(dbscan_img, dsize=(256 * k, 171 * k))
        # _, h_labels = find_clusters(dbscan_img, t_min_cluster_pixel_count, t_max_cluster_pixel_count)
        # tok = time.time()

        # print(f"Found {len(h_labels)} clusters in {tok - tik} seconds")

        # Benchmark cv2 thresholding
        tik = time.time()
        masked = mask_flake_color(img0, flake_avg_hsv)
        tok = time.time()

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

        for cnt in contours:
            if cv2.contourArea(cnt) < t_min_cluster_pixel_count:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("threshold", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

        baseline = cv2.imread(f"C:\\04_03_23_EC_1\\MLScanned2\\TileScan_001--Stage{s}.jpg")
        cv2.imshow("threshold", baseline)
        cv2.waitKey()

        print(f"Finished contour search in {tok - tik} seconds")
        print("-----")
