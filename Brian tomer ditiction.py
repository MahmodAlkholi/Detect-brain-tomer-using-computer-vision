import cv2
import numpy as np

# Load the image
img = cv2.imread('brain_tumor.jpeg')

# Reshape the image to a 2D array of pixels and 3 color values
pixel_values = img.reshape((-1, 3))

# Convert the pixel values to float32
pixel_values = np.float32(pixel_values)

# Define the criteria for the k-means algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set the number of clusters (segments) to 4
k = 4

# Perform k-means clustering
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert the centers of each cluster to integers
centers = np.uint8(centers)

# Reshape the labels to the original image shape
labels = labels.flatten()
segmented_image = centers[labels.flatten()]

# Reshape the segmented image to the original image shape
segmented_image = segmented_image.reshape(img.shape)

# Display the original and segmented images
cv2.imshow('Original Image', img)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
