from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import depthai as dai
from object_detector import *

app = Flask(__name__)

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# Load Object Detector
detector = HomogeneousBgDetector()

def compute_integral_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the image
    height, width = gray_image.shape

    # Initialize integral image
    integral_image = np.zeros((height, width), dtype=np.uint32)

    # Compute integral image manually
    for y in range(height):
        for x in range(width):
            if y == 0 and x == 0:
                integral_image[y, x] = gray_image[y, x]
            elif y == 0:
                integral_image[y, x] = integral_image[y, x - 1] + gray_image[y, x]
            elif x == 0:
                integral_image[y, x] = integral_image[y - 1, x] + gray_image[y, x]
            else:
                integral_image[y, x] = (integral_image[y - 1, x] +
                                         integral_image[y, x - 1] -
                                         integral_image[y - 1, x - 1] +
                                         gray_image[y, x])

    # Normalize the integral image for display
    integral_image_normalized = ((integral_image - integral_image.min()) / (integral_image.max() - integral_image.min()) * 255).astype(np.uint8)

    # Convert integral image to colored format for display
    integral_image_colored = cv2.cvtColor(integral_image_normalized, cv2.COLOR_GRAY2BGR)

    return integral_image_colored

def image_stitch(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors in both images
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Initialize feature matcher
    matcher = cv2.BFMatcher()

    # Match descriptors between the two images
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Minimum number of matches required for homography estimation
    MIN_MATCH_COUNT = 10
    if len(good_matches) < MIN_MATCH_COUNT:
        print("Insufficient matches found.")
        return None

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Check if homography estimation failed
    if H is None:
        print("Homography estimation failed.")
        return None

    # Warp image1 to image2 using homography
    warped_image = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))

    # Blend the warped image with image2
    result = warped_image.copy()
    result[:image2.shape[0], :image2.shape[1]] = image2

    return result

# Function to process the DepthAI camera frames
def process_frames():
    # Pipeline initialization
    pipeline = dai.Pipeline()

    # Define the DepthAI camera
    camRgb = pipeline.create(dai.node.ColorCamera)
    xout = pipeline.create(dai.node.XLinkOut)

    xout.setStreamName('video')

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(30)

    # Linking
    camRgb.video.link(xout.input)

    # Start the pipeline
    with dai.Device(pipeline) as device:

        # Output queue will be used to get the video frames from the output defined above
        q = device.getOutputQueue(name="video", maxSize=30, blocking=True)

        while True:
            # Get DepthAI camera frame
            frame = q.get().getCvFrame()

            # Compute integral image
            integral_image = compute_integral_image(frame)

            # Stitch images
            stitched_image = image_stitch(frame, frame)  # Replace the second 'frame' with another image if needed

            # Convert frame to JPEG for HTML display
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for image upload and stitching
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if 'image1' and 'image2' are in the request
        if 'image1' in request.files and 'image2' in request.files:
            # Get the uploaded images from the request
            image1 = request.files['image1']
            image2 = request.files['image2']

            # Read the images
            image1_np = cv2.imdecode(np.fromstring(image1.read(), np.uint8), cv2.IMREAD_COLOR)
            image2_np = cv2.imdecode(np.fromstring(image2.read(), np.uint8), cv2.IMREAD_COLOR)

            # Perform image stitching
            stitched_image = image_stitch(image1_np, image2_np)

            # Convert stitched image to JPEG for HTML display
            _, jpeg = cv2.imencode('.jpg', stitched_image)
            stitched_image_bytes = jpeg.tobytes()

            return Response(b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + stitched_image_bytes + b'\r\n',
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        elif 'image1' in request.files:
            # Get the uploaded image from the request
            image1 = request.files['image1']

            # Read the image
            image1_np = cv2.imdecode(np.fromstring(image1.read(), np.uint8), cv2.IMREAD_COLOR)

            # Compute the integral image for the single image
            integral_image = compute_integral_image(image1_np)

            # Convert integral image to JPEG for HTML display
            _, jpeg = cv2.imencode('.jpg', integral_image)
            integral_image_bytes = jpeg.tobytes()

            return Response(b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + integral_image_bytes + b'\r\n',
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return "No images uploaded"


# Route for integral image computation
@app.route('/integral_feed', methods=['POST'])
def integral_feed():
    if request.method == 'POST':
        # Get the uploaded images from the request
        image1 = request.files['image1']

        # Read the images
        image1_np = cv2.imdecode(np.fromstring(image1.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Compute integral image for each image
        integral_image1 = compute_integral_image(image1_np)
        
        # Convert integral images to JPEG for HTML display
        _, jpeg1 = cv2.imencode('.jpg', integral_image1)
        integral_image_bytes1 = jpeg1.tobytes()

        return Response(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + integral_image_bytes1 + b'\r\n',
                        mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True)
