"""
Note: Currently only configured for Exfoliator tilescans. Very unlikely to work well on other datasets.
"""
import argparse
import glob
import os
import re
import time
from multiprocessing import Pool

import cv2
import numpy as np
import matplotlib

from util.config import load_config
from util.leica import dim_get, pos_get
from util.plot import make_plot, location
from util.processing import bg_to_flake_color, get_avg_rgb, find_clusters, edgefind
from util.box import merge_boxes, Box
from util.logger import logger


flake_colors_rgb = [
    # # Thick-looking
    # [6, 55, 94],
    # Monolayer-looking
    # [57, 65, 86],
    # [60, 66, 85],
    # [89,99,109],
    [0, 0, 0],
]
flake_colors_hsv = [
    np.uint8(matplotlib.colors.rgb_to_hsv(x) * np.array([179, 255, 255])) for x in flake_colors_rgb
    # matplotlib outputs in range 0,1. Opencv expects HSV images in range [0,0,0] to [179, 255,255]
]
flake_color_hsv = np.mean(flake_colors_hsv, axis=0)
avg_rgb = np.mean(flake_colors_rgb, axis=0)

output_dir = 'cv_output'
threadsave = 1  # number of threads NOT allocated when running
boundflag = 1
t_rgb_dist = 8
# t_hue_dist = 12 #12
t_red_dist = 12
# t_red_cutoff = 0.1 #fraction of the chunked image that must be more blue than red to be binned
t_color_match_count = 0.000225  # fraction of image that must look like monolayers
k = 4
t_min_cluster_pixel_count = 30 * (k / 4) ** 2  # flake too small
t_max_cluster_pixel_count = 20000 * (k / 4) ** 2  # flake too large
# scale factor for DB scan. recommended values are 3 or 4. Trade-off in time vs accuracy. Impact epsilon.
scale = 1  # the resolution images are saved at, relative to the original file. Does not affect DB scan


def run_file(img_filepath, output_dir, scan_pos_dict, dims):
    tik = time.time()

    try:
        img0 = cv2.imread(img_filepath)
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        pixcal = 1314.08 / w  # microns/pixel from Leica calibration
        pixcals = [pixcal, 876.13 / h]

        lowlim = np.array([87, 100, 99]) # defines lower limit for what code can see as background
        highlim = np.array([114, 118, 114])

        imsmall = cv2.resize(img.copy(), dsize=(256 * k, 171 * k)).reshape(-1, 3)
        test = np.sign(imsmall - lowlim) + np.sign(highlim - imsmall)

        # chooses pixels between provided limits, quickly filtering to potential background pixels
        pixout = imsmall * np.sign(test + abs(test))
        if len(pixout) == 0:  # making sure background is identified
            # print('Pixel failed')
            return

        # Get monolayer color from background color, and calculate distance between each pixel and predicted flake RGB
        back_rgb = get_avg_rgb(pixout)
        flake_avg_rgb = bg_to_flake_color(back_rgb)

        img_pixels = img.copy().reshape(-1, 3)
        rgb_pixel_dists = np.sqrt(np.sum((img_pixels - flake_avg_rgb) ** 2, axis=1))

        # Mask the image to only the pixels close enough to predicted flake color
        img_mask = np.logical_and(rgb_pixel_dists < t_rgb_dist, back_rgb[0] - img_pixels[:, 0] > 5)

        # If the number of close pixels is under the threshold, return early
        t_count = np.sum(img_mask)
        if t_count < t_color_match_count * len(img_pixels):
            # print('Count failed', t_count)
            return

        logger.debug(f"{img_filepath} meets count thresh with {t_count}")

        # If there are too many dark pixels in the image, the image is likely at the edge of the scan; return early
        pixdark = np.sum((img_pixels[:, 2] < 25) * (img_pixels[:, 1] < 25) * (img_pixels[:, 0] < 25))
        if np.sum(pixdark) / len(img_pixels) > 0.1:
            logger.debug(f"{img_filepath} was on an edge!")
            return

        # Create Masked image
        img2_mask_in = img.copy().reshape(-1, 3)
        img2_mask_in[~img_mask] = np.array([0, 0, 0])
        img2_mask_in = img2_mask_in.reshape(img.shape)

        # DB SCAN, fitting to find clusters of correctly colored pixels
        dbscan_img = cv2.cvtColor(img2_mask_in, cv2.COLOR_RGB2GRAY)
        dbscan_img = cv2.resize(dbscan_img, dsize=(256 * k, 171 * k))

        labels, h_labels = find_clusters(dbscan_img, t_min_cluster_pixel_count, t_max_cluster_pixel_count)

        if len(h_labels) < 1:
            return
        logger.debug(f"{img_filepath} had {len(h_labels)} filtered dbscan clusters")

        # Make boxes
        boxes = []
        for label_id in h_labels:
            # Find bounding box... in x/y plane find min value. This is just argmin and argmax
            criteria = labels == label_id
            criteria = criteria.reshape(dbscan_img.shape[:2]).astype(np.uint8)
            x = np.where(criteria.sum(axis=0) > 0)[0]
            y = np.where(criteria.sum(axis=1) > 0)[0]
            width = x.max() - x.min()
            height = y.max() - y.min()
            boxes.append(Box(label_id, x.min(), y.min(), width, height))

        # Merge boxes that overlap
        pass_one = merge_boxes(dbscan_img, boxes)
        merged_boxes = merge_boxes(dbscan_img, pass_one)

        if not merged_boxes:
            return

        # Make patches out of clusters
        wantshape = (int(int(img.shape[1]) * scale), int(int(img.shape[0]) * scale))
        bscale = wantshape[0] / (256 * k)  # need to scale up box from dbscan image
        offset = 5
        patches = [
            [int((int(b.x) - offset) * bscale), int((int(b.y) - offset) * bscale),
             int((int(b.width) + 2 * offset) * bscale), int((int(b.height) + 2 * offset) * bscale)] for b
            in merged_boxes
        ]
        logger.debug('patched')
        color = (0, 0, 255)
        thickness = 6
        log_file = open(output_dir + "Color Log.txt", "a+")

        stage = int(re.search(r"Stage(\d{3})", img_filepath).group(1))
        imloc = location(stage, dims)

        # Convert back from (x, y) scan number to mm coordinates
        radius = 1
        i = -1
        while radius > 0.1:
            i = i + 1
            radius = (int(imloc[0]) - int(scan_pos_dict[i][0])) ** 2 + (int(imloc[1]) - int(scan_pos_dict[i][2])) ** 2
        posx = scan_pos_dict[i][1]
        posy = scan_pos_dict[i][3]
        posstr = "X:" + str(round(1000 * posx, 2)) + ", Y:" + str(round(1000 * posy, 2))

        img0 = cv2.putText(img0, posstr, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
        img4 = img0.copy()

        for p in patches:
            logger.debug(p)
            y_min = int(p[0] + 2 * offset * bscale / 3)
            y_max = int(p[0] + p[2] - 2 * offset * bscale / 3)
            x_min = int(p[1] + 2 * offset * bscale / 3)
            x_max = int(p[1] + p[3] - 2 * offset * bscale / 3)  # note that the offsets cancel here
            logger.debug((x_min, y_min, x_max, y_max))
            bounds = [max(0, p[1]), min(p[1] + p[3], int(h)), max(0, p[0]), min(p[0] + p[2], int(w))]
            imchunk = img[bounds[0]:bounds[1], bounds[2]:bounds[3]]  # identifying bounding box of flake

            width = round(p[2] * pixcal, 1)
            height = round(p[3] * pixcal, 1)  # microns

            # creating the output images
            img3 = cv2.rectangle(img0, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), color, thickness)
            img3 = cv2.putText(img3, str(height), (p[0] + p[2] + 10, p[1] + int(p[3] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 0, 0), 2, cv2.LINE_AA)
            img3 = cv2.putText(img3, str(width), (p[0], p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                               cv2.LINE_AA)

            flake_rgb = [0, 0, 0]
            farea = 0
            if boundflag:
                flake_rgb, edgeim, farea = edgefind(imchunk, flake_avg_rgb, pixcals, t_rgb_dist)  # calculating border pixels
                print('Edge found')
                img4 = cv2.rectangle(img4, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), color, thickness)
                img4 = cv2.putText(img4, str(height), (p[0] + p[2] + 10, p[1] + int(p[3] / 2)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img4 = cv2.putText(img4, str(width), (p[0], p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                                   cv2.LINE_AA)
                img4[bounds[0]:bounds[1], bounds[2]:bounds[3]] = img4[bounds[0]:bounds[1], bounds[2]:bounds[3]] + edgeim

            logstr = str(stage) + ',' + str(farea) + ',' + str(flake_rgb[0]) + ',' + str(flake_rgb[1]) + ',' + str(
                flake_rgb[2]) + ',' + str(back_rgb[0]) + ',' + str(back_rgb[1]) + ',' + str(back_rgb[2])
            log_file.write(logstr + '\n')

        log_file.close()

        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_filepath)), img3)
        if boundflag:
            cv2.imwrite(
                os.path.join(output_dir + "\\AreaSort\\", str(int(farea)) + '_' + os.path.basename(img_filepath)), img4)
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
        files = [
            [f, output_dir, scanposdict, dims] for f in input_files if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"]
        ]

        n_proc = os.cpu_count() - threadsave  # config.jobs if config.jobs > 0 else
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

            f.write('flake_colors_rgb=' + str(flake_colors_rgb) + '\n')
            f.write('t_rgb_dist=' + str(t_rgb_dist) + '\n')
            # f.write('t_hue_dist='+str(t_hue_dist)+'\n')
            f.write('t_red_dist=' + str(t_red_dist) + '\n')
            # f.write('t_red_cutoff='+str(t_red_cutoff)+'\n')
            f.write('t_color_match_count=' + str(t_color_match_count) + '\n')
            f.write('t_min_cluster_pixel_count=' + str(t_min_cluster_pixel_count) + '\n')
            f.write('t_max_cluster_pixel_count=' + str(t_max_cluster_pixel_count) + '\n')
            f.write('k=' + str(k) + "\n\n")

        flist = open(output_dir + "Imlist.txt", "w+")
        flist.write("List of Stage Numbers for copying to Analysis Sheet" + "\n")
        flist.close()
        flist = open(output_dir + "Imlist.txt", "a+")
        fwrite = open(output_dir + "By Area.txt", "w+")
        fwrite.write("Num, A" + "\n")
        fwrite.close()
        fwrite = open(output_dir + "By Area.txt", "a+")

        stages = np.sort(np.array([int(re.search(r"Stage(\d{3})", file).group(1)) for file in output_files]))
        make_plot(stages, dims, output_dir)  # creating cartoon for file
        flist.close()

        # print(output_dir+"Color Log.txt")
        N, A, Rf, Gf, Bf, Rw, Gw, Bw = np.loadtxt(output_dir + "Color Log.txt", skiprows=1, delimiter=',', unpack=True)

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
