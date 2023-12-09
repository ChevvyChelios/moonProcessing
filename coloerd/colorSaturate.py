import cv2
import numpy as np

def enhance_saturation(image, saturation_factor=1.5):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Multiply the saturation channel by the specified factor
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)

    # Convert the image back to BGR color space
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return enhanced_image

# Load an image of the moon with different geological features
# image_path = "colorImage/EnEnhac.png"
image_path = "colorImage/EnhanchedImage.png"
moon_image = cv2.imread(image_path)

# Enhance saturation to make colors more vivid
saturation_factor_value = 1.5  # Increase for more vivid colors, decrease for less
enhanced_moon_image = enhance_saturation(moon_image, saturation_factor=saturation_factor_value)

# Display the original and enhanced images
cv2.imshow('Original Moon Image', moon_image)
cv2.imshow('Enhanced Moon Image', enhanced_moon_image)

# cv2.imwrite("colorImage/saturatedImage.png", enhanced_moon_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
