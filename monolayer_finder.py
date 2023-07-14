"""
Note: Currently only configured for Exfoliator tilescans. Very unlikely to work well on other datasets.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

from util.config import load_config
from util.leica import dim_get, pos_get, get_stage
from util.plot import make_plot, location
from util.processing import bg_to_flake_color, get_avg_rgb, mask_flake_color, apply_morph_open, apply_morph_close
from util.box import merge_boxes, Box
from util.logger import logger


threadsave = 1  # number of threads NOT allocated when running
boundflag = 1
# t_color_match_count = 0.000225  # fraction of image that must look like monolayers
k = 4
t_min_cluster_pixel_count = 1500  # flake too small
# t_max_cluster_pixel_count = 20000 * (k / 4) ** 2  # flake too large


def run_file(img_filepath, output_dir, scan_pos_dict, dims):
    tik = time.time()

    try:
        stage = get_stage(img_filepath)

        img0 = cv2.imread(img_filepath)
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        pixcal = 1314.08 / w  # microns/pixel from Leica calibration
        # pixcals = [pixcal, 876.13 / h]

        # Lower and higher RGB limits for what code can see as background
        lowlim = np.array([87, 100, 99])
        highlim = np.array([114, 118, 114])

        # chooses pixels between provided limits, quickly filtering to potential background pixels
        start = time.time()
        imsmall = cv2.resize(img.copy(), dsize=(256 * k, 171 * k)).reshape(-1, 3)
        test = np.sign(imsmall - lowlim) + np.sign(highlim - imsmall)
        pixout = imsmall * np.sign(test + abs(test))
        end = time.time()

        logger.debug(f"Stage{stage} background detection in {end - start} seconds")

        if len(pixout) == 0:  # making sure background is identified
            return

        # Get monolayer color from background color
        back_rgb = get_avg_rgb(pixout)
        flake_avg_rgb = bg_to_flake_color(back_rgb)
        flake_avg_hsv = cv2.cvtColor(np.uint8([[flake_avg_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?

        # TODO: necessary?
        img_pixels = img.copy().reshape(-1, 3)

        # If there are too many dark pixels in the image, the image is likely at the edge of the scan; return early
        start = time.time()
        pixdark = np.sum((img_pixels[:, 2] < 25) * (img_pixels[:, 1] < 25) * (img_pixels[:, 0] < 25))

        if np.sum(pixdark) / len(img_pixels) > 0.1:
            logger.debug(f"Stage{stage} was on an edge!")
            return

        end = time.time()
        logger.debug(f"Stage{stage} tested for dark pixels in {end - start} seconds")

        # Mask image using thresholds and apply morph operations to reduce false positives
        start = time.time()
        masked = mask_flake_color(img0, flake_avg_hsv)
        dst = apply_morph_open(masked)
        dst = apply_morph_close(dst)
        end = time.time()

        logger.debug(f"Stage{stage} thresholded and transformed in {end - start} seconds")

        # Find contours of masked and processed image
        start = time.time()
        contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        end = time.time()

        if len(contours) < 1:
            return
        logger.debug(f"Stage{stage} had {len(contours)} contours in {end - start} seconds")

        # Make boxes
        start = time.time()

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            if area < t_min_cluster_pixel_count:
                continue
            boxes.append(Box(cnt, area, x, y, w, h))

        end = time.time()
        logger.debug(f"Stage{stage} generated boxes in {end - start} seconds")

        # Merge boxes that overlap
        start = time.time()
        merged_boxes = merge_boxes(masked, boxes)
        merged_boxes = merge_boxes(masked, merged_boxes)
        end = time.time()

        logger.debug(f"Stage{stage} merged boxes in {end - start} seconds")

        if not merged_boxes:
            return

        log_file = open(output_dir + "Color Log.txt", "a+")
        xd, yd, _, _ = location(stage, dims)

        # Convert back from (x, y) scan number to mm coordinates
        try:
            posy, posx = scan_pos_dict[int(yd), int(xd)]
            pos_str = "X:" + str(round(1000 * posx, 2)) + ", Y:" + str(round(1000 * posy, 2))
        except IndexError:
            logger.warn(f'Stage{stage} pos conversion failed!')
            pos_str = ""

        # Label output images
        start = time.time()
        color = (0, 0, 255)
        thickness = 6
        font = cv2.FONT_HERSHEY_SIMPLEX

        img0 = cv2.putText(img0, pos_str, (100, 100), font, 3, (0, 0, 0), 2, cv2.LINE_AA)
        img4 = img0.copy()

        offset = 5
        max_area = 0

        for b in merged_boxes:
            offset_x = int(b.x) - offset
            offset_y = int(b.y) - offset
            offset_w = int(b.width) + 2 * offset
            offset_h = int(b.height) + 2 * offset

            width_microns = round(offset_w * pixcal, 1)
            height_microns = round(offset_h * pixcal, 1)  # microns

            max_area = max(int(b.area), max_area)
            logger.debug((offset_x + offset_w + 10, offset_y + int(offset_h / 2)))

            # creating the output images
            img3 = cv2.rectangle(img0, (offset_x, offset_y), (offset_x + offset_w, offset_y + offset_h), color, thickness)
            img3 = cv2.putText(img3, str(height_microns), (offset_x + offset_w + 10, offset_y + int(offset_h / 2)), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
            img3 = cv2.putText(img3, str(width_microns), (offset_x, offset_y - 10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

            if boundflag:
                logger.debug('Drawing contour bounds...')
                img4 = cv2.rectangle(img4, (offset_x, offset_y), (offset_x + offset_w, offset_y + offset_h), color, thickness)
                img4 = cv2.putText(img4, str(height_microns), (offset_x + offset_w + 10, offset_y + int(offset_h / 2)), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img4 = cv2.putText(img4, str(width_microns), (offset_x, offset_y - 10), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img4 = cv2.drawContours(img4, b.contours, -1, (0, 0, 255), 2)

            log_str = str(stage) + ',' + str(b.area) + ',' + str(back_rgb[0]) + ',' + str(back_rgb[1]) + ',' + str(back_rgb[2])
            log_file.write(log_str + '\n')

        log_file.close()
        end = time.time()

        logger.debug(f"Stage{stage} labelled images in {end - start} seconds")

        start = time.time()
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_filepath)), img3)

        if boundflag:
            cv2.imwrite(os.path.join(output_dir + "\\AreaSort\\", str(max_area) + '_' + os.path.basename(img_filepath)), img4)

        end = time.time()
        logger.debug(f"Stage{stage} saved images in {end - start} seconds")

    except Exception as e:
        logger.warn(f"Exception occurred: {e}")

    tok = time.time()
    logger.info(f"{img_filepath} - {tok - tik} seconds")


def main(args):
    config = load_config(args.q)

    for input_dir, output_dir in config:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + "\\AreaSort\\", exist_ok=True)

        input_files = [f for f in glob.glob(os.path.join(input_dir, "*")) if "Stage" in f]
        input_files.sort(key=len)

        with open(output_dir + "Color Log.txt", "w+") as log_file:
            log_file.write('N,A,Rf,Gf,Bf,Rw,Gw,Bw\n')

        tik = time.time()
        scanposdict = pos_get(input_dir)
        dims = dim_get(input_dir)

        n_proc = os.cpu_count() - threadsave
        files = [
            [f, output_dir, scanposdict, dims] for f in input_files
            if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"]
        ]

        with Pool(n_proc) as pool:
            pool.starmap(run_file, files)

        tok = time.time()

        output_files = [
            f for f in glob.glob(os.path.join(output_dir, "*"))
            if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"] and "Stage" in f
        ]
        filecount = len(output_files)

        with open(output_dir + "Summary.txt", "a+") as f:
            f.write(f"Total for {len(files)} files: {tok - tik} = avg of {(tok - tik) / len(files)} per file on {n_proc} logical processors\n")
            f.write(str(filecount) + ' identified flakes\n')

            f.write('t_min_cluster_pixel_count=' + str(t_min_cluster_pixel_count) + '\n')
            f.write('k=' + str(k) + "\n\n")

        flist = open(output_dir + "Imlist.txt", "w+")
        flist.write("List of Stage Numbers for copying to Analysis Sheet" + "\n")
        flist.close()
        flist = open(output_dir + "Imlist.txt", "a+")
        flist.close()  # TODO: what is the purpose of this?

        fwrite = open(output_dir + "By Area.txt", "w+")
        fwrite.write("Num, A" + "\n")
        fwrite.close()
        fwrite = open(output_dir + "By Area.txt", "a+")

        start = time.time()
        stages = np.sort(np.array([get_stage(file) for file in output_files]))
        make_plot(stages, dims, output_dir)  # creating cartoon for file
        end = time.time()

        logger.info(f"Created coordmap.jpg in {end - start} seconds")

        # print(output_dir+"Color Log.txt")
        N, A, Rw, Gw, Bw = np.loadtxt(output_dir + "Color Log.txt", skiprows=1, delimiter=',', unpack=True)

        pairs = []
        i = 0
        while i < len(A):
            pair = np.array([N[i], A[i]])
            pairs.append(pair)
            i = i + 1
        # print(pairs)
        pairsort = sorted(pairs, key=lambda x: x[1], reverse=True)
        # print(pairs,pairsort)
        for pair in pairsort:
            writestr = str(int(pair[0])) + ', ' + str(pair[1]) + '\n'
            fwrite.write(writestr)
        fwrite.close()

        logger.info(f"Total for {len(files)} files: {tok - tik} = avg of {(tok - tik) / len(files)} per file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find graphene flakes on SiO2. Currently configured only for exfoliator dataset"
    )
    parser.add_argument(
        "--q",
        type=str,
        default="Queue.txt",
        help="Directory containing images to process. Optional unless running in headless mode"
    )
    args = parser.parse_args()
    main(args)
