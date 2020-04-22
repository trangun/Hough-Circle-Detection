import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.filters import gaussian
from skimage import io
from scipy import ndimage as ndi
import cv2


# Load picture and detect edges
img = io.imread('./circle.png')
image_gray = color.rgb2gray(img)

im = ndi.gaussian_filter(image_gray, 1.5, mode="reflect")

# blur = gaussian(image_gray,0.2)
im = img_as_ubyte(im[0:260, 70:280])
edges = canny(im, sigma=2, low_threshold=0.55, high_threshold=0.8)

# Detect two radii
hough_radii = np.arange(25, 75, 2)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 4 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=4)

# Draw them
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
ax.set_title('Detected Circles', fontsize=50) # title of plot
image = color.gray2rgb(img[0:260, 70:280])
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
    image[circy, circx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()