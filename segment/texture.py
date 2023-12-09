
import cv2
import numpy as np
from skimage.filters import gabor
from skimage import color
import matplotlib.pyplot as plt

def texture_segmentation(image, frequency, theta):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gabor filter for texture analysis
    gabor_response, _ = gabor(gray_image, frequency=frequency, theta=theta)

    # Threshold the Gabor response to obtain a binary mask
    binary_mask = gabor_response > np.percentile(gabor_response, 90)

    # Apply the binary mask to the original image to obtain the segmented result
    segmented_image = cv2.bitwise_and(image, image, mask=binary_mask.astype(np.uint8))

    return segmented_image

# Load an image of the moon
image_path = "colorImage/EnEnhac.png"
moon_image = cv2.imread(image_path)

# Perform texture-based segmentation using Gabor filters
frequency_value = 0.7  # Adjust this value based on the desired texture frequency
theta_value = 0.0  # Adjust this value based on the desired orientation of the Gabor filter
segmented_moon_image = texture_segmentation(moon_image, frequency=frequency_value, theta=theta_value)

# Display the original and segmented images
cv2.imshow('Original Moon Image', moon_image)
cv2.imshow('Texture-based Segmentation', segmented_moon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
