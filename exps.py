import cv2
import numpy as np
from matplotlib import pyplot as plt

# # Read the image
# image = cv2.imread('./imgg.jpg')
#
# # Split the image into its color channels
# b, g, r = cv2.split(image)
#
# # Apply GaussianBlur to each color channel
# b_blurred = cv2.GaussianBlur(b, (3, 3), 0)
# g_blurred = cv2.GaussianBlur(g, (3, 3), 0)
# r_blurred = cv2.GaussianBlur(r, (3, 3), 0)
#
# # Create a sharpening filter
# kernel = np.array([[-1, -1, -1],
#                    [-1,  9, -1],
#                    [-1, -1, -1]])
#
# # Apply the filter to each color channel to sharpen them
# b_sharpened = cv2.filter2D(b_blurred, -1, kernel)
# g_sharpened = cv2.filter2D(g_blurred, -1, kernel)
# r_sharpened = cv2.filter2D(r_blurred, -1, kernel)
#
# # Merge the sharpened color channels back into an RGB image
# sharpened_image_rgb = cv2.merge([b_sharpened, g_sharpened, r_sharpened])
#
# # Save the sharpened image
# cv2.imwrite('sharpened_image_rgb.jpg', sharpened_image_rgb)
#
# # Display the images
# plt.figure(figsize=(12, 4))
# plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(132), plt.imshow(cv2.cvtColor(sharpened_image_rgb, cv2.COLOR_BGR2RGB)), plt.title('Sharpened Image (RGB)')
# plt.subplot(133), plt.imshow(cv2.cvtColor(image - sharpened_image_rgb, cv2.COLOR_BGR2RGB)), plt.title('Difference')
# plt.show()
