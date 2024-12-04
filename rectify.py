import cv2
import numpy as np
import os


def get_kp_SIFT(imgL, imgR):
    """
    SIFT detector & FLANN-based matcher to get keypoints, descritpors, and matches
    """
    detector = cv2.SIFT_create(nfeatures = 1024, sigma = 1.6)
    kp1, des1 = detector.detectAndCompute(imgL, None)
    kp2, des2 = detector.detectAndCompute(imgR, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    return kp1, des1, kp2, des2, knn_matches


def get_kp_ORB(imgL, imgR):
    """
    ORB detector & FLANN-based matcher to get keypoints, descritpors, and matches
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)
    
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,
        key_size=12,
        multi_probe_level=1,
    )
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = matcher.knnMatch(des1, des2, k=2) # keep 2 closest matches
    return kp1, des1, kp2, des2, knn_matches


def lowes_ratio_test(matches, ratio_threshold=0.6):
    """
    LRT checks if matches can be discerned from random noise. If it can't, it's not a good match.
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


def draw_matches(imgL, imgR, kp1, des1, kp2, des2, flann_match_pairs, n=8):
    """Draw the first n mathces between the left and right images."""
    # https://docs.opencv.org/4.2.0/d4/d5d/group__features2d__draw.html
    # https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
    img = cv2.drawMatches(
        imgL,
        kp1,
        imgR,
        kp2,
        flann_match_pairs[:n],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    # cv2.imshow("Matches", img)
    cv2.imwrite("OUT/SIFT_Matches.png", img)
    # cv2.waitKey(0)


def compute_fundamental_matrix(matches, kp1, kp2, method=cv2.FM_RANSAC):
    """
    Estimate the fundamental matrix using good matches
    https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm
    """
    pts1, pts2 = [], []
    fundamental_matrix, inliers = None, None
    # get the 16 best matches
    for m in matches: # matches[:] + 
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    if pts1 and pts2:
        fundamental_matrix, inliers = cv2.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
            ransacReprojThreshold=0.5,
            confidence=0.9999,
        )
    return fundamental_matrix, inliers, pts1, pts2


def get_fundamental_matrix(imgL, imgR, debug=False):
    """
    Get the fundamental matrix using SIFT keypoints, LRT and RANSAC
    """
    kp1, des1, kp2, des2, flann_match_pairs = get_kp_SIFT(imgL, imgR)
    good_matches = lowes_ratio_test(flann_match_pairs, 0.2)
    if debug:
        draw_matches(imgL, imgR, kp1, des1, kp2, des2, good_matches)

    F, I, points1, points2 = compute_fundamental_matrix(good_matches, kp1, kp2)
    return F, I, points1, points2


def correct_distortion(imgL, imgR, H1, H2, w1, h1, w2, h2, imgPath, debug=False, additional=""):
    """
    Correct the distortion in the images using the homography matrices
    """
    imgL_undistorted = cv2.warpPerspective(imgL, H1, (w1, h1))
    imgR_undistorted = cv2.warpPerspective(imgR, H2, (w2, h2))
    if debug:
        if not os.path.exists("OUT/DEBUG/"+imgPath):
            os.makedirs("OUT/DEBUG/"+imgPath)
        cv2.imwrite("OUT/DEBUG/"+imgPath+"/"+additional+"undistorted_L.png", imgL_undistorted)
        # cv2.imwrite("OUT/undistorted_R.png", imgR_undistorted)
    return imgL_undistorted, imgR_undistorted


def rectify_images(imgL, imgR, F, points1, points2, imgPath="", debug=False):
    """
    Rectify the images using the fundamental matrix and the homography matrices
    """
    h1, w1 = imgL.shape
    h2, w2 = imgR.shape
    thresh = 1
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(points1), np.float32(points2), F, imgSize=(w1, h1), threshold=thresh,
    )
    imgL_undistorted, imgR_undistorted = correct_distortion(
        imgL, imgR, H1, H2, w1, h1, w2, h2, imgPath, debug, additional="rectified_"
    )
    return imgL_undistorted, imgR_undistorted