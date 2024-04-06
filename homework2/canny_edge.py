import cv2 as cv
import numpy as np

def gkernel(size=3, sig=2):
    """\
    Gaussian Kernel Creator via given length and sigma
    """

    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)

class CannyEdgeDetector:
    def __init__(self, image_path):
        self.image_path = image_path

    def detect_edges(self):
        # Read the image
        image = cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)

        # Gaussian blur kernel
        gaussian = gkernel(3,2)

        # Smooth the image
        smoothed_image = cv.filter2D(image, -1, gaussian)

        # Compute gradients using Sobel operator
        sobel_x = cv.Sobel(smoothed_image, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(smoothed_image, cv.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude and direction
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)

        # Non-maximum suppression
        edges = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                direction = gradient_direction[i, j]
                mag = gradient_magnitude[i, j]
                if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                    if (mag > gradient_magnitude[i, j + 1]) and (mag > gradient_magnitude[i, j - 1]):
                        edges[i, j] = mag
                elif (22.5 <= direction < 67.5):
                    if (mag > gradient_magnitude[i + 1, j - 1]) and (mag > gradient_magnitude[i - 1, j + 1]):
                        edges[i, j] = mag
                elif (67.5 <= direction < 112.5):
                    if (mag > gradient_magnitude[i + 1, j]) and (mag > gradient_magnitude[i - 1, j]):
                        edges[i, j] = mag
                elif (112.5 <= direction < 157.5):
                    if (mag > gradient_magnitude[i + 1, j + 1]) and (mag > gradient_magnitude[i - 1, j - 1]):
                        edges[i, j] = mag

        # Apply double thresholding
        high_threshold = 0.2 * np.max(edges)
        low_threshold = 0.1 * np.max(edges)
        strong_edges = (edges > high_threshold)
        weak_edges = (edges > low_threshold) & (edges <= high_threshold)

        # Edge tracking by hysteresis
        strong_edges_final = cv.dilate(strong_edges.astype(np.uint8), None)
        weak_edges_final = weak_edges.astype(np.uint8)
        _, thresh = cv.threshold(strong_edges_final, 0, 255, cv.THRESH_BINARY)
        edges_final = cv.bitwise_and(weak_edges_final, thresh)

        return edges_final




# # Example usage
# detector = CannyEdgeDetector("image.jpg")
# edges = detector.detect_edges()
# plt.imshow(edges, cmap='gray')
# plt.title('Canny Edges')
# plt.axis('off')
# plt.show()
