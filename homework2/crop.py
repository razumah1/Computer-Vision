
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

image = cv2.imread(r'./frame_479.png',cv2.IMREAD_GRAYSCALE)

Image.fromarray(image, 'L').save('./org.png')
cropped = np.array(image[442:447, 1088:1093])
#cropped = np.array(image[:480,1780:1785])
Image.fromarray(cropped, 'L').save('./cropped.png')
#image[2145:2150,3750:3755] = 255
#image[92:397,218:1000] = 255
print(cropped)
print(np.average(cropped))
plt.figure()
plt.subplot(1, 2, 1)
plt.axis('off')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.axis('off')
plt.imshow(cropped, cmap='gray')
plt.plot()
