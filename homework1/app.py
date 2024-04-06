from flask import Flask, render_template, Response
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

            # Get Aruco marker
            corners, _, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
            if corners:

                # Draw polygon around the marker
                int_corners = np.int0(corners)
                cv2.polylines(frame, int_corners, True, (0, 255, 0), 5)

                # Aruco Perimeter
                aruco_perimeter = cv2.arcLength(corners[0], True)

                # Pixel to cm ratio
                pixel_cm_ratio = aruco_perimeter / 20

                contours = detector.detect_objects(frame)

                # Draw objects boundaries
                for cnt in contours:
                    # Get rect
                    rect = cv2.minAreaRect(cnt)
                    (x, y), (w, h), angle = rect

                    # Get Width and Height of the Objects by applying the Ratio pixel to cm
                    object_width = w / pixel_cm_ratio
                    object_height = h / pixel_cm_ratio

                    # Display rectangle
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.polylines(frame, [box], True, (255, 0, 0), 2)
                    cv2.putText(frame, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                    cv2.putText(frame, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

            # Convert frame to JPEG for HTML display
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

