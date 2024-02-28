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


import pandas as pd

# Sample DataFrame
data = {'song': ['Song1', 'Song2', 'Song3', 'Song4', 'Song5', 'Song6', 'Song7'],
        'artist': ['Artist1', 'Artist1', 'Artist1', 'Artist2', 'Artist2', 'Artist2', 'Artist2']}
df = pd.DataFrame(data)

# Count the occurrences of each artist
artist_counts = df['artist'].value_counts()

# Filter the DataFrame to include only the rows where the artist appears more than 2 times
popular_artists = artist_counts[artist_counts > 2].index
popular_artists
# Filter the DataFrame based on popular artists using .loc[]
popular_artist_df = df.loc[df['artist'].isin(popular_artists)]

print(popular_artist_df)

# Example DataFrame
data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}
df = pd.DataFrame(data)
import pandas as pd
# Function to calculate mean of other cells in the same column
# Function to calculate mean of other cells in the same column
def calculate_mean_other_cells(row):
    column_name = row.name
    other_cells = df[column_name].drop(row.name)
    return other_cells.mean()

# Apply the function to each cell in the DataFrame
means_df = df.apply(calculate_mean_other_cells, axis=1)

print(means_df)