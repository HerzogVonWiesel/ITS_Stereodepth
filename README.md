# ITS_Stereodepth

Repository for the Fall 2024 project of Intelligent Transportation Systems.

## Authors

- Jerome Chirol
- Marvin Jerome Stephan

This repository is divided in two parts: /ITS_Midas_depth, which contains the code and analysis done by Jerome Chirol, and /ITS_Stereodepth, which contains the code and analysis done by Marvin Jerome Stephan.

## ITS_Stereodepth

This repository presents a pipeline for depth estimation from stereo images, focusing on both calibrated and uncalibrated setups. Implemented using traditional computer vision techniques, the pipeline avoids the computational demands of deep learning models while enabling flexibility for amateur stereo configurations.
The algorithm uses image rectification, disparity computa- tion using Semi-Global Block Matching (SGBM), and enhances through Weighted Least Squares (WLS) filtering. The pipeline also supports metric depth calculation when camera parameters are given. Its performance is evaluated against Apple’s Depth Pro model and ground truth disparities from an established dataset.
Results underline the pipeline’s adaptability, especially in a tested DIY setup, but also its limitations in handling stark rectification and achieving precise metric depth.

The main script (ITS_Stereodepth/stereo_test.py) supports several commandline options for flexibility:
1) path [required]: Specifies the path to the input files’ folder.
2) –skiprectify (-s): Skips the rectification steps if the input images are already rectified.
3) –rectifyfile (-r): Specifies the path to a known camera geometry file for calibrated instead of uncalibrated rectifica- tion.
4) –debug (-d): Enables debug mode, providing detailed outputs and visualizations for troubleshooting.
5) –output (-o): Sets the output folder for saving the results.
6) –metric (-m): Enables metric depth calculation by spec- ifying the focal length (in mm), sensor width (in mm), and baseline distance (the distance between the two cameras in meters).

The following command serves as an example of how to run the script. It processes the input folder IN/moto (where our left image and right image are saved), skips rectification, enables debug mode, and calculates metric depth using a focal length of 40 mm, a sensor width of 30 mm, and a baseline of 0.193 meters:

`python stereo_test.py IN/moto -d -s -m 40 30 0.193`