# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

image = cv2.imread(r'./image_2.png', cv2.IMREAD_GRAYSCALE)
Image.fromarray(image, 'L').save('./org.png')
cropped = np.array(image[377:382,930:935])
#cropped = np.array(image[:776,230:1006])
Image.fromarray(cropped, 'L').save('./cropped.png')
# image[1777:1782,1750:1755] = 255
image[92:397,218:1000] = 255
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
