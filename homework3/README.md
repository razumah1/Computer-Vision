Welcome to the Computer Vision Project repository! This repository contains Python scripts for various computer vision tasks implemented using OpenCV library. Below is a detailed overview of each script along with the necessary dependencies and instructions on how to run them.

Contents:
Question 1: Image Matching with SSD

This script implements image matching using the Sum of Squared Differences (SSD) algorithm and a sliding window technique. It compares a reference image with a dataset of images to find similar regions.
Question 3: Stereo Vision Depth Estimation

The script performs stereo vision processing to estimate depth from a pair of stereo images captured by a stereo camera setup. It calculates the fundamental matrix, rectifies the images, and computes the depth map.
Question 4: Optical Flow Motion Detection

This script computes optical flow and visualizes motion vectors in a video sequence. It uses the Farneback method to estimate motion between consecutive frames.
Question 5: Keypoint Detection and Matching with SIFT

The script detects keypoints and extracts features from images using the Scale-Invariant Feature Transform (SIFT) algorithm. It then matches keypoints between images to find correspondences.
Question 6: Object Recognition with BoVW and SVM

This script performs object recognition using the Bag-of-Visual-Words (BoVW) approach. It clusters keypoints into visual words using KMeans clustering and trains an SVM classifier for recognition.
Question 7: Fundamental Matrix Computation and Rectification

The script computes the fundamental matrix between a pair of stereo images and rectifies them for stereo vision applications.
Question 8: Object Detection and Tracking with ArUco Markers and Haar Cascade Classifier

Integration with Web Application:

The app.py file integrates the project with a web application.
Run app.py and click the IP address to be redirected to the webpage.

This script detects and tracks objects using ArUco markers for identification and a Haar cascade classifier for face detection.
Dependencies:
Python 3.x
OpenCV (cv2)
NumPy
Matplotlib
scikit-learn (for Question 6)
Instructions:
Each script can be run independently by executing the Python file in a compatible environment.
Ensure that the necessary dependencies such as OpenCV and NumPy are installed.
Some scripts may require additional resources such as image or video files, which should be provided in the appropriate directories.
Modify the scripts as needed to suit specific use cases or datasets.
Additional Notes:
Please refer to the comments within each script for detailed explanations of the implementation steps.
For any questions or issues, feel free to contact the repository owner.
