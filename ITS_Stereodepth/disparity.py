import cv2
import os


def stereo_sgbm(imgL_undistorted, imgR_undistorted, imgPath, debug=False, additional=""):
    """
    Compute the disparity map using stereo Semi-Global Block Matching
    """
    # Trial & Error disparity parameters
    win_size = 2
    min_disp = 0
    max_disp = 280
    num_disp = max_disp - min_disp
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=5,
        uniquenessRatio=15,
        speckleWindowSize=5,
        speckleRange=5,
        disp12MaxDiff=2,
        P1=8 * 3 * win_size ** 2,
        P2=32 * 3 * win_size ** 2,
    )
    disparity_SGBM_l = stereo.compute(imgL_undistorted, imgR_undistorted)
    # Using StereoBM
    # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    # disparity_BM = stereo.compute(imgL_undistorted, imgR_undistorted)

    # plt.imshow(disparity_SGBM, "gray")
    # plt.colorbar()
    # plt.show()
    if debug:
        cv2.imwrite(os.path.join(imgPath, "DEBUG/"+additional+"_depth_noisy.png"), \
            cv2.normalize(src=disparity_SGBM_l, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX))

    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    disparity_SGBM_r = right_matcher.compute(imgR_undistorted, imgL_undistorted)
    return disparity_SGBM_l, disparity_SGBM_r, stereo


def wls_filter(disparity_SGBM, disparity_SGBM_r, imgL_undistorted, stereo, imgPath):
    """
    Apply the Weighted Least Squares filter to the disparity map to smooth it out
    """
    # https://docs.opencv.org/4.2.0/d3/d14/tutorial_ximgproc_disparity_filtering.html
    sigma = 1.5
    lmbda = 8000.0
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(disparity_SGBM, imgL_undistorted, disparity_map_right=disparity_SGBM_r)

    # plt.imshow(filtered_disp, "gray")
    # plt.colorbar()
    # plt.show()
    # filtered_disp = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    # print(path+additional+"depth_map_filtered.png")
    # cv2.imwrite("OUT/"+path+"/"+additional+"depth_map_filtered.png", filtered_disp)
    return filtered_disp


def depth_process(imgL_undistorted, imgR_undistorted, imgPath, debug=False, additional=""):
    """
    Get filtered depth map from (rectified) stereo images
    """
    disparity_SGBM_l, disparity_SGBM_r, stereo = stereo_sgbm(imgL_undistorted, imgR_undistorted, imgPath, debug, additional)
    filtered_disp = wls_filter(disparity_SGBM_l, disparity_SGBM_r, imgL_undistorted, stereo, imgPath)
    # filtered_disp = disparity_SGBM_l
    return filtered_disp