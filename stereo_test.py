import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
# imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)


def get_keypoints_and_descriptors(imgL, imgR):
    """Use ORB detector and FLANN matcher to get keypoints, descritpors,
    and corresponding matches that will be good for computing
    homography.
    """
    #orb = cv2.ORB_create()
    #kp1, des1 = orb.detectAndCompute(imgL, None)
    #kp2, des2 = orb.detectAndCompute(imgR, None)
    minHessian = 400
    detector = cv2.SIFT_create(nfeatures = 1024, sigma = 1.6) # (hessianThreshold=minHessian)
    kp1, des1 = detector.detectAndCompute(imgL, None)
    kp2, des2 = detector.detectAndCompute(imgR, None)
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SIFT is a floating-point descriptor NORM_L2 is used
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)
    return kp1, des1, kp2, des2, knn_matches

    # Each keypoint of the first image is matched with a number of
    # keypoints from the second image. k=2 means keep the 2 best matches
    # for each keypoint (best matches = the ones with the smallest
    # distance measurement).
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann_match_pairs = flann.knnMatch(des1, des2, k=2)
    return kp1, des1, kp2, des2, flann_match_pairs


def lowes_ratio_test(matches, ratio_threshold=0.6):
    """Filter matches using the Lowe's ratio test.

    The ratio test checks if matches are ambiguous and should be
    removed by checking that the two distances are sufficiently
    different. If they are not, then the match at that keypoint is
    ignored.

    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    """
    filtered_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            filtered_matches.append(m)
    # sort the matches based on distance
    # filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)[::-1]
    print(len(filtered_matches))
    return filtered_matches


def draw_matches(imgL, imgR, kp1, des1, kp2, des2, flann_match_pairs):
    """Draw the first 8 mathces between the left and right images."""
    # https://docs.opencv.org/4.2.0/d4/d5d/group__features2d__draw.html
    # https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
    img = cv2.drawMatches(
        imgL,
        kp1,
        imgR,
        kp2,
        flann_match_pairs[:8],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    # cv2.imshow("Matches", img)
    cv2.imwrite("OUT/SIFT_Matches.png", img)
    # cv2.waitKey(0)


def compute_fundamental_matrix(matches, kp1, kp2, method=cv2.FM_RANSAC):
    """Use the set of good mathces to estimate the Fundamental Matrix.

    See  https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm
    for more info.
    """
    pts1, pts2 = [], []
    fundamental_matrix, inliers = None, None
    # get the 16 best matches
    for m in matches: # matches[:] + 
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    if pts1 and pts2:
        # You can play with the Threshold and confidence values here
        # until you get something that gives you reasonable results. I
        # used the defaults
        fundamental_matrix, inliers = cv2.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
            ransacReprojThreshold=0.5,
            confidence=0.9999,
        )
    return fundamental_matrix, inliers, pts1, pts2

# get imgL and imgR from videos


isVideo = False
debug = True
path = "motorcycle"
suffixes = ["_l", "_r"]
img_format = ".png"

additional = "00001" if isVideo else ""
#print(path+additional+suffixes[0]+img_format)
imgL = cv2.imread(path+additional+suffixes[0]+img_format, cv2.IMREAD_GRAYSCALE)  # left image
imgR = cv2.imread(path+additional+suffixes[1]+img_format, cv2.IMREAD_GRAYSCALE)  # right image

kp1, des1, kp2, des2, flann_match_pairs = get_keypoints_and_descriptors(imgL, imgR)
good_matches = lowes_ratio_test(flann_match_pairs, 0.2)
if debug:
    draw_matches(imgL, imgR, kp1, des1, kp2, des2, good_matches)


F, I, points1, points2 = compute_fundamental_matrix(good_matches, kp1, kp2)

h1, w1 = imgL.shape
h2, w2 = imgR.shape
thresh = 1
_, H1, H2 = cv2.stereoRectifyUncalibrated(
    np.float32(points1), np.float32(points2), F, imgSize=(w1, h1), threshold=thresh,
)

def correct_distortion(imgL, imgR, H1, H2, w1, h1, w2, h2, debug, additional=""):
    imgL_undistorted = cv2.warpPerspective(imgL, H1, (w1, h1))
    imgR_undistorted = cv2.warpPerspective(imgR, H2, (w2, h2))
    if debug:
        if not os.path.exists("OUT/DEBUG/"+path):
            os.makedirs("OUT/DEBUG/"+path)
        cv2.imwrite("OUT/DEBUG/"+path+"/"+additional+"undistorted_L.png", imgL_undistorted)
        # cv2.imwrite("OUT/undistorted_R.png", imgR_undistorted)
    return imgL_undistorted, imgR_undistorted

def depth_process(imgL_undistorted, imgR_undistorted, debug, additional=""):
    # if output path doesnt exist create it
    if not os.path.exists("OUT/DEBUG/"+path):
        os.makedirs("OUT/DEBUG/"+path)

    if not os.path.exists("OUT/"+path):
        os.makedirs("OUT/"+path)
    # ### TODO: -DEBUG- ###
    # imgL_undistorted = imgL
    # imgR_undistorted = imgR

    # Using StereoBM
    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # disparity_BM = stereo.compute(imgL_undistorted, imgR_undistorted)
    # plt.imshow(disparity_BM, "gray")
    # plt.colorbar()
    # plt.show()


    # Using StereoSGBM
    # Set disparity parameters. Note: disparity range is tuned according to
    #  specific parameters obtained through trial and error.
    win_size = 2
    min_disp = 0
    max_disp = 256
    num_disp = max_disp - min_disp  # Needs to be divisible by 16
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=5,
        uniquenessRatio=5,
        speckleWindowSize=5,
        speckleRange=5,
        disp12MaxDiff=2,
        P1=8 * 3 * win_size ** 2,
        P2=32 * 3 * win_size ** 2,
    )
    disparity_SGBM = stereo.compute(imgL_undistorted, imgR_undistorted)
    # plt.imshow(disparity_SGBM, "gray")
    # plt.colorbar()
    # plt.show()
    if debug:
        if not os.path.exists("OUT/DEBUG/"+path):
            os.makedirs("OUT/DEBUG/"+path)
        cv2.imwrite("OUT/DEBUG/"+path+"/"+additional+"depth_map_noisy.png", cv2.normalize(src=disparity_SGBM, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX))

    # use WLS filtering to get better depth map
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    disparity_SGBM_r = right_matcher.compute(imgR_undistorted, imgL_undistorted)
    sigma = 1.5
    lmbda = 8000.0
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(disparity_SGBM, imgL_undistorted, disparity_map_right=disparity_SGBM_r)

    # plt.imshow(filtered_disp, "gray")
    # plt.colorbar()
    # plt.show()
    # save without losing quality (CV_16U)
    filtered_disp = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    print(path+additional+"depth_map_filtered.png")
    cv2.imwrite("OUT/"+path+"/"+additional+"depth_map_filtered.png", filtered_disp)


if isVideo:
    videoLength = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])/2
    for i in range(1, int(videoLength)+1):
        additional = str(i).zfill(5)
        imgL = cv2.imread(path+additional+suffixes[0]+img_format, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(path+additional+suffixes[1]+img_format, cv2.IMREAD_GRAYSCALE)
        imgL_undistorted, imgR_undistorted = correct_distortion(imgL, imgR, H1, H2, w1, h1, w2, h2, debug)
        depth_process(imgL_undistorted, imgR_undistorted, debug, additional)
else:
    imgL_undistorted, imgR_undistorted = correct_distortion(imgL, imgR, H1, H2, w1, h1, w2, h2, debug)
    ## TODO: DEBUG TESTING
    imgL_undistorted, imgR_undistorted = imgL, imgR
    depth_process(imgL_undistorted, imgR_undistorted, debug)