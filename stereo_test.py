import cv2
import os

from rectify import get_fundamental_matrix, rectify_images
from disparity import depth_process


isVideo = False
debug = True
path = "phone2"
suffixes = ["_l", "_r"]
img_format = ".jpg"

additional = "00001" if isVideo else ""
#print(path+additional+suffixes[0]+img_format)
imgL = cv2.imread(path+additional+suffixes[0]+img_format, cv2.IMREAD_GRAYSCALE)  # left image
imgR = cv2.imread(path+additional+suffixes[1]+img_format, cv2.IMREAD_GRAYSCALE)  # right image

def compute_save_depth(imgL_undistorted, imgR_undistorted, path, normalize=True, debug=False, additional=""):
    depth_map = depth_process(imgL_undistorted, imgR_undistorted, path, debug, additional)
    if normalize:
        depth_map = cv2.normalize(src=depth_map, dst=depth_map, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    # print(path+additional+"depth_map_filtered.png")
    cv2.imwrite("OUT/"+path+"/"+additional+"depth_map_filtered.png", depth_map)

F, I, points1, points2 = get_fundamental_matrix(imgL, imgR, debug)
if isVideo:
    videoLength = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])/2
    for i in range(1, int(videoLength)+1):
        additional = str(i).zfill(5)
        imgL = cv2.imread(path+additional+suffixes[0]+img_format, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(path+additional+suffixes[1]+img_format, cv2.IMREAD_GRAYSCALE)
        imgL_undistorted, imgR_undistorted = rectify_images(imgL, imgR, F, points1, points2, path, debug)
        compute_save_depth(imgL_undistorted, imgR_undistorted, path, normalize=True, debug=debug, additional=additional)
else:
    imgL_undistorted, imgR_undistorted = rectify_images(imgL, imgR, F, points1, points2, path, debug)
    ## TODO: DEBUG TESTING
    # imgL_undistorted, imgR_undistorted = imgL, imgR
    compute_save_depth(imgL_undistorted, imgR_undistorted, path, normalize=True, debug=debug, additional=additional)