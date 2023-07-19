import cv2
import numpy as np


threadsave = 1  # number of threads NOT allocated when running
boundflag = 1
# t_color_match_count = 0.000225  # fraction of image that must look like monolayers
k = 4
t_min_cluster_pixel_count = 1500
# t_max_cluster_pixel_count = 20000 * (k / 4) ** 2  # flake too large
flake_angle_tolerance_rads = np.deg2rad(10)

# Morphology parameters
open_morph_size = 2
close_morph_size = 3

open_morph_shape = cv2.MORPH_RECT
close_morph_shape = cv2.MORPH_CROSS

# Labelling parameters
box_offset = 5
box_color = (255, 0, 0)  # tuple of (R, G, B)
box_thickness = 6
font = cv2.FONT_HERSHEY_SIMPLEX
