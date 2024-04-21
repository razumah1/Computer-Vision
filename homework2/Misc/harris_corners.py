# %%
import cv2 as cv
import numpy as np

class HarrisCornerDetector:
    def __init__(self, image_path):
        self.image_path = image_path

    def detect_corners(self):
        # Read the image
        image = cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)

        # Gaussian blur kernel
        gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

        # Smooth the image
        image_padded = np.full((7, 7), np.average(image), dtype=np.float32)
        image_padded[1:6, 1:6] = image
        smooth = cv.filter2D(image_padded, -1, gaussian)[1:6, 1:6]

        # Compute gradients using Sobel operator
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        grad_x = np.array(cv.filter2D(smooth, -1, sobel_x), dtype=np.float32)[1:6, 1:6]
        grad_y = np.array(cv.filter2D(smooth, -1, sobel_y), dtype=np.float32)[1:6, 1:6]

        # Compute elements of the auto-correlation matrix
        ixx = grad_x * grad_x
        ixy = grad_x * grad_y
        iyy = grad_y * grad_y

        # Smooth the elements of the auto-correlation matrix
        ixx_padded = np.full((7, 7), np.average(ixx), dtype=np.float32)
        ixx_padded[1:6, 1:6] = ixx
        ixy_padded = np.full((7, 7), np.average(ixy), dtype=np.float32)
        ixy_padded[1:6, 1:6] = ixy
        iyy_padded = np.full((7, 7), np.average(iyy), dtype=np.float32)
        iyy_padded[1:6, 1:6] = iyy

        # Gaussian blur the auto-correlation matrices
        sxx = cv.filter2D(ixx_padded, -1, gaussian)[1:6, 1:6]
        sxy = cv.filter2D(ixy_padded, -1, gaussian)[1:6, 1:6]
        syy = cv.filter2D(iyy_padded, -1, gaussian)[1:6, 1:6]

        # Compute Harris corner response
        R = np.zeros((5, 5), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                M = np.array([[sxx[i, j], sxy[i, j]],
                              [sxy[i, j], syy[i, j]]], dtype=np.float32)
                lambdas = np.linalg.eigvals(M)
                R[i, j] = (lambdas[0] * lambdas[1]) - 0.05 * ((lambdas[0] + lambdas[1]) ** 2)

        return R

if __name__ == "__main__":
    detector = HarrisCornerDetector("/Users/reza/School/2024/1-Spring/Computer Vision/Assignment 2/2/patch.png")
    corners = detector.detect_corners()
    print(corners)


# %%


