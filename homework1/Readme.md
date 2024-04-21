Camera Calibration and Object Dimension Measurement
1. Camera Calibration
Calibration Matrix
The calibration matrix for the chosen camera needs to be determined. This matrix includes parameters such as focal length, principal point, and distortion coefficients. To verify the calibration matrix, an example involving imaging a known set of reference points, such as a chessboard pattern, can be used.

2. Image Formation Pipeline Equation and Parameter Estimation
Intrinsic and Extrinsic Parameters
Using the image formation pipeline equation, a series of linear equations in matrix form can be set up to solve for intrinsic and extrinsic parameters. This involves capturing a series of images while changing the orientation of the camera in each iteration. One of these images is selected, and measurements of actual 3D world points are taken, along with corresponding pixel coordinates. By solving the linear equations, intrinsic parameters (related to the camera itself) and extrinsic parameters (related to the camera's position and orientation) can be estimated. Rotation matrices can be computed from the extrinsic parameters, along with the angles of rotation along each axis.

3. Object Dimension Measurement Using Perspective Projection Equations
A script can be written to find the real-world dimensions of an object using perspective projection equations. This involves imaging an object using the calibrated camera from a specific distance and measuring the object's pixel coordinates. By applying perspective projection equations, the real-world dimensions of the object can be computed.

4. Web Application Implementation
Application Overview
An application is developed to compute real-world dimensions of an object in view. The application runS as a web application on a browser and is OS agnostic.

Implementation Details
The application is implemented using web technologies such as HTML, and python flask. It utilizes the camera access API to capture images from the user's camera. OpenCV.js or similar libraries can be used for image processing and computation of object dimensions. Justifiable assumptions are made regarding the identification of points of interest on the object. To run the web application, you will need to run the "app.py" file and click the ip address to access the website.

Object Size Measurement using DepthAI and OpenCV
Overview
This Python script utilizes the DepthAI library along with OpenCV to measure the size of objects detected in a camera frame. The script detects Aruco markers as reference points and uses them to calculate the size of objects in the scene. The size is displayed in centimeters on the video feed.

Requirements
Python 3.x
DepthAI library
OpenCV library
Numpy library
Object Detector module (imported as HomogeneousBgDetector)
Setup
Install Python 3.x if not already installed.

Install the required libraries using pip:

Copy code
pip install depthai opencv-python numpy
Ensure that the object_detector.py file containing the HomogeneousBgDetector class is available in the same directory or accessible through Python's module search path.

Usage
Run the script by executing the Python file.

Copy code
python script_name.py
A window will open displaying the camera feed. Aruco markers and detected objects will be outlined, with their respective dimensions displayed in centimeters.

Press the 'q' key to exit the application.

Customization
The script can be modified to use a different camera resolution or frame rate by adjusting the parameters in the camRgb.setResolution() and camRgb.setFps() methods, respectively.
Additional object detection or image processing techniques can be integrated into the script to enhance its functionality.
Dependencies
DepthAI: DepthAI is an embedded platform built to enable depth perception in applications. It provides tools for running neural networks and processing camera data efficiently.
OpenCV: OpenCV is a popular computer vision library used for various image processing tasks, including object detection and manipulation.
Numpy: Numpy is a fundamental package for scientific computing with Python, used for numerical operations and array manipulation.
