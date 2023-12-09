
import cv2
import numpy as np

def segment_moon_features(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for different features (adjust these values based on your image)
    crater_lower = np.array([0, 0, 50])
    crater_upper = np.array([180, 50, 255])

    mountain_lower = np.array([0, 0, 100])
    mountain_upper = np.array([180, 50, 255])

    plains_lower = np.array([0, 0, 120])
    plains_upper = np.array([180, 50, 255])

    # Create binary masks for each feature
    crater_mask = cv2.inRange(hsv_image, crater_lower, crater_upper)
    mountain_mask = cv2.inRange(hsv_image, mountain_lower, mountain_upper)
    plains_mask = cv2.inRange(hsv_image, plains_lower, plains_upper)

    # Combine the masks to create a final segmented image
    segmented_image = cv2.bitwise_or(image, image, mask=crater_mask)
    segmented_image = cv2.bitwise_or(segmented_image, segmented_image, mask=mountain_mask)
    segmented_image = cv2.bitwise_or(segmented_image, segmented_image, mask=plains_mask)

    return segmented_image

# Load an image of the moon
image_path = "colorImage/EnEnhac.png"
moon_image = cv2.imread(image_path)

# Perform semantic segmentation of moon features
segmented_moon_features = segment_moon_features(moon_image)

# Display the original and segmented images
cv2.imshow('Original Moon Image', moon_image)
cv2.imshow('Segmented Moon Features', segmented_moon_features)
cv2.waitKey(0)
cv2.destroyAllWindows()
