import cv2
import numpy as np

def enhance_details(image, sigma=1.5, strength=1.5, brightness=1.0, contrast=1.0):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply unsharp masking to the L channel
    l_channel_enhanced = cv2.GaussianBlur(l_channel, (0, 0), sigma)
    l_channel_enhanced = cv2.addWeighted(l_channel, 1.0 + strength, l_channel_enhanced, -strength, 0)

    # Merge the enhanced L channel with the original A and B channels
    enhanced_lab_image = cv2.merge([l_channel_enhanced, a_channel, b_channel])

    # Convert the enhanced LAB image back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    # Adjust brightness and contrast
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=contrast, beta=brightness)

    return enhanced_image

# Load a colored image
image_path = "colorImage/EnhanchedImage.png"
original_image = cv2.imread(image_path)

# Adjust details, brightness, and contrast
sigma_value = 1.5  # Adjust this value based on the level of details you want
strength_value = 1.5  # Adjust this value for the strength of the enhancement
brightness_value = 1.2  # Adjust this value for increased brightness
contrast_value = 1.2  # Adjust this value for increased contrast

enhanced_image = enhance_details(original_image, sigma=sigma_value, strength=strength_value,
                                  brightness=brightness_value, contrast=contrast_value)

# im = cv2.imread('./moon.jpg')
# cv2.imshow('rereal', im)


# Display the original and enhanced images
cv2.imshow('Original Image', original_image)
cv2.imshow('Enhanced Image', enhanced_image)

cv2.imwrite("colorImage/EnEnhac.png", enhanced_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
