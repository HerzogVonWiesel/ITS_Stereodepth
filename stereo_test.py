import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import cv2
import argparse
import numpy as np
import json

from rectify import get_fundamental_matrix, rectify_images, rectify_images_calibrated
from disparity import depth_process


parser = argparse.ArgumentParser(description="A script to rectify stereo images and compute the disparity map.")
parser.add_argument("path", type=str, help="Path to the input file (required).")
parser.add_argument("-s", "--skiprectify", action="store_true", help="Skip rectification (optional).")
parser.add_argument("-r", "--rectifyfile", type=str, help="Path to the rectification file (optional).")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode (optional).")
parser.add_argument("-o", "--output", type=str, help="Path to the output folder (optional).")
parser.add_argument("-m", "--metric", nargs=2, type=float, help="Focal length & sensor width to enable metric depth map (EXPERIMENTAL).")

args = parser.parse_args()


def compute_save_depth(imgL_undistorted, imgR_undistorted, path, metric, debug=False, baseline=1.0, additional=""):
    depth_map = depth_process(imgL_undistorted, imgR_undistorted, path, debug, additional)
    if not metric:
        depth_map = cv2.normalize(src=depth_map, dst=depth_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    else:
        focal_mm, sensor_width_mm = metric
        # (focal_mm / sensor_width_mm) * image_width_in_pixels
        focal_pixel = (focal_mm / sensor_width_mm) * imgL_undistorted.shape[1]
        print(f"Focal length in pixels: {focal_pixel}")
        # baseline (meter) * focal (pixel) / disparity (pixel)
        depth_map = baseline * focal_pixel / depth_map
        depth_map = np.float32(depth_map)
    # print one value nearly at the top left corner, one value in the center, and one value nearly at the bottom right corner
    if debug:
        print(f"Depth map values: {depth_map[0, 0]}, {depth_map[depth_map.shape[0]//2, depth_map.shape[1]//2]}, {depth_map[-1, -1]}")
    img_format = ".exr" if metric else ".png"
    cv2.imwrite(os.path.join(path, "OUT/"+additional+"_depth"+img_format), depth_map)


input_images = [(os.path.join(args.path, name), name.split('.')[0]) for name in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, name))]
if len(input_images) < 2:
    print("Error: Not enough images in the directory.")
    exit(1)

input_images.sort()
print(f"Processing {len(input_images)//2} stereo pairs.")
# print(input_images)

if args.output:
    if args.debug:
        if not os.path.exists(os.path.join(args.output, "DEBUG")):
            os.makedirs(os.path.join(args.output, "DEBUG"))
    if not os.path.exists(os.path.join(args.output, "OUT")):
        os.makedirs(os.path.join(args.output, "OUT"))
else:
    if args.debug:
        if not os.path.exists(os.path.join(args.path, "DEBUG")):
            os.makedirs(os.path.join(args.path, "DEBUG"))
    if not os.path.exists(os.path.join(args.path, "OUT")):
        os.makedirs(os.path.join(args.path, "OUT"))
    args.output = args.path

baseline = 1.0
if args.rectifyfile:
    with open(args.rectifyfile, "r") as f:
        data = json.load(f)
        cam0 = np.array(data["cam0"])
        cam1 = np.array(data["cam1"])
        baseline = data["baseline"]
elif not args.skiprectify:
    F, I, points1, points2 = get_fundamental_matrix(cv2.imread(input_images[0][0], cv2.IMREAD_GRAYSCALE), cv2.imread(input_images[1][0], cv2.IMREAD_GRAYSCALE), args.debug, args.output, additional=input_images[0][1])

for i in range(len(input_images)//2):
    imgL = cv2.imread(input_images[i*2][0], cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(input_images[i*2+1][0], cv2.IMREAD_GRAYSCALE)
    if args.rectifyfile:
        imgL, imgR = rectify_images_calibrated(imgL, imgR, cam0, cam1, args.debug)
        if args.debug:
            cv2.imwrite(os.path.join(args.output, "DEBUG/"+input_images[i*2][1]+"_undistorted.png"), imgL)
    elif not args.skiprectify:
        imgL, imgR = rectify_images(imgL, imgR, F, points1, points2, args.output, args.debug, additional=input_images[i*2][1])
    compute_save_depth(imgL, imgR, args.output, args.metric, debug=args.debug, baseline=baseline, additional=input_images[i*2][1])