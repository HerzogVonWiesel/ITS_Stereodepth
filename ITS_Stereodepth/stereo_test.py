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
parser.add_argument("-m", "--metric", nargs=3, type=float, help="Focal length (mm), sensor width (mm) and distance between cameras (m) to enable metric depth map (EXPERIMENTAL).")

args = parser.parse_args()


def compute_save_depth(imgL_undistorted, imgR_undistorted, path, metric, baseline=0.2, debug=False, additional=""):
    depth_map = depth_process(imgL_undistorted, imgR_undistorted, path, debug, additional)
    # depth_map = np.right_shift(depth_map, 4)
    # depth_map = np.float32(depth_map) / 16
    print(f"Depth map type: {depth_map.dtype}")
    if not metric:
        depth_map = cv2.normalize(src=depth_map, dst=depth_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    else:
        print(f"Depth map values: {depth_map[405, 505]}, {depth_map[depth_map.shape[0]//2, depth_map.shape[1]//2]}, {depth_map[-1, -1]}")
        focal_mm, sensor_width_mm, baseline = metric
        # (focal_mm / sensor_width_mm) * image_width_in_pixels
        focal_pixel = (focal_mm / sensor_width_mm) * imgL_undistorted.shape[1]
        print(f"Focal length in pixels: {focal_pixel}")
        # baseline (meter) * focal (pixel) / disparity (pixel)
        # baseline += 1
        depth_map = baseline * focal_pixel / (depth_map +131)
        depth_map = np.float32(depth_map)
    # print one value nearly at the top left corner, one value in the center, and one value nearly at the bottom right corner
    if debug:
        # save image with red square around the points
        print(f"Depth map values: {depth_map[405, 505]}, {depth_map[depth_map.shape[0]//2, depth_map.shape[1]//2]}, {depth_map[-1, -1]}")
        # cv2.rectangle(depth_map, (480, 480), (520, 520), (255, 0, 0), 20)
        # cv2.rectangle(depth_map, (depth_map.shape[1]//2-20, depth_map.shape[0]//2-20), (depth_map.shape[1]//2+20, depth_map.shape[0]//2+20), (255, 0, 0), 20)
        # cv2.imwrite(os.path.join(path, "DEBUG/"+additional+"_depth.png"), depth_map)
    img_format = ".exr" if metric else ".png"
    cv2.imwrite(os.path.join(path, "OUT/"+additional+"_depth"+img_format), depth_map)


# depth_map = cv2.imread(args.path, cv2.IMREAD_UNCHANGED)
# print(f"Depth map values: {depth_map[405, 505]}, {depth_map[depth_map.shape[0]//2, depth_map.shape[1]//2]}, {depth_map[-1, -1]}")
# focal_mm, sensor_width_mm, baseline = 40, 30, 0.193
# # (focal_mm / sensor_width_mm) * image_width_in_pixels
# focal_pixel = (focal_mm / sensor_width_mm) * depth_map.shape[1]
# print(f"Focal length in pixels: {focal_pixel}")
# # baseline (meter) * focal (pixel) / disparity (pixel)
# # baseline += 1
# # depth_map = baseline * focal_pixel / (depth_map + 131)
# # depth_map = np.int16(depth_map / 32)
# print(f"Depth map values: {depth_map[405, 505]}, {depth_map[depth_map.shape[0]//2, depth_map.shape[1]//2]}, {depth_map[-1, -1]}")
# # depth_map = cv2.normalize(src=depth_map, dst=depth_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
# cv2.rectangle(depth_map, (depth_map.shape[1]//2-20, depth_map.shape[0]//2-20), (depth_map.shape[1]//2+20, depth_map.shape[0]//2+20), (255, 255, 255), 20)
# cv2.rectangle(depth_map, (480, 480), (520, 520), (255, 255, 255), 20)
# cv2.imwrite(os.path.join("IN/test/", "OUT/depthpro.png"), depth_map)
# print(f"Depth map values: {depth_map[405, 505]}, {depth_map[depth_map.shape[0]//2, depth_map.shape[1]//2]}, {depth_map[-1, -1]}")
# exit(0)

input_images = [(os.path.join(args.path, name), name.split('.')[0]) for name in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, name)) and name.endswith(('.jpg', '.png', '.jpeg'))]
if len(input_images) < 2:
    print("Error: Not enough images in the directory.")
    exit(1)

input_images.sort()
# print(f"input_images: {input_images}")
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

baseline = 0.2
if args.rectifyfile:
    with open(args.rectifyfile, "r") as f:
        data = json.load(f)
        cam0 = np.array(data["cam0"])
        cam1 = np.array(data["cam1"])
        baseline = data["baseline"]
elif not args.skiprectify:
    imgL = cv2.imread(input_images[0][0], cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(input_images[1][0], cv2.IMREAD_GRAYSCALE)
    # resize to 1/4th of the original size
    # imgL = cv2.resize(imgL, (0, 0), fx=0.33, fy=0.33)
    # imgR = cv2.resize(imgR, (0, 0), fx=0.33, fy=0.33)
    F, I, points1, points2 = get_fundamental_matrix(imgL, imgR, args.debug, args.output, additional=input_images[0][1])

for i in range(len(input_images)//2):
    imgL = cv2.imread(input_images[i*2][0], cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(input_images[i*2+1][0], cv2.IMREAD_GRAYSCALE)
    # print(f"Processing stereo pair {input_images[i*2][1]} and {input_images[i*2+1][1]}.")
    # resize to 1/4th of the original size
    # imgL = cv2.resize(imgL, (0, 0), fx=0.5, fy=0.5)
    # imgR = cv2.resize(imgR, (0, 0), fx=0.5, fy=0.5)
    if args.rectifyfile:
        imgL, imgR = rectify_images_calibrated(imgL, imgR, cam0, cam1, args.debug)
        if args.debug:
            cv2.imwrite(os.path.join(args.output, "DEBUG/"+input_images[i*2][1]+"_undistorted.png"), imgL)
    elif not args.skiprectify:
        imgL, imgR = rectify_images(imgL, imgR, F, points1, points2, args.output, args.debug, additional=input_images[i*2][1])
    compute_save_depth(imgL, imgR, args.output, args.metric, debug=args.debug, additional=input_images[i*2][1])