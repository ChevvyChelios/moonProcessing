
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries

def superpixel_segmentation(image, num_segments=100, compactness=10):
    # Convert the image to a floating-point format
    image_float = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    # Perform SLIC superpixel segmentation
    segments = slic(image_float, n_segments=num_segments, compactness=compactness)

    # Create a visual representation of the superpixels
    boundaries = mark_boundaries(image_float, segments)

    # Convert the visual representation back to the BGR color space
    boundaries = (boundaries * 255).astype(np.uint8)

    return boundaries

# Load an image
image_path = "colorImage/EnEnhac.png"
original_image = cv2.imread(image_path)

# Perform superpixel segmentation
num_segments_value = 100  # Adjust based on the desired number of superpixels
compactness_value = 10  # Adjust based on the compactness of the segments
superpixel_result = superpixel_segmentation(original_image, num_segments=num_segments_value, compactness=compactness_value)

# Display the original image and superpixel segmentation result
cv2.imshow('Original Image', original_image)
cv2.imshow('Superpixel Segmentation Result', superpixel_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
