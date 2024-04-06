import cv2
from object_detector import *
import numpy as np


# Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# # Load object detector
detector = HomogeneousBgDetector()

# Load Cap
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:

    _, img = cap.read()
    color_img = img.copy()

    #Get Aruco marker
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    if corners:
        
        # Draw polygon around the marker
        int_corners = np.intp(corners)
        cv2.polylines(color_img, int_corners, True, (0, 255, 0), 5)

        # Aruko Perimeter
        aruco_perimeter = cv2.arcLength(corners[0], True)
        # print(aruco_perimeter) #the value that's printed out is the corrsponding pixel value for 20 cm, we can use this ratio to calculate object length 

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20 # 1 cm = 29.52677 pixel


        contours = detector.detect_objects(img)
        # print(contours)

        # Draw object boundaries
        for cnt in contours:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get width and height of the object by applying the pixel to cm ratio
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

