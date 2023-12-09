
import cv2
import numpy as np

def apply_color_filter(image, lower_bound, upper_bound):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a mask using the specified color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    return filtered_image

# Load an image of the moon
image_path = "colorImage/saturatedImage.png"
moon_image = cv2.imread(image_path)

# Define color range for a specific wavelength or color on the moon's surface (e.g., blue tones)
lower_color_bound = np.array([5,17,23])  # Adjust these values based on the desired color range
upper_color_bound = np.array([105,147,200])

# Apply color filter to emphasize specific wavelengths or colors
filtered_moon_image = apply_color_filter(moon_image, lower_bound=lower_color_bound, upper_bound=upper_color_bound)

# Display the original and filtered images
cv2.imshow('Original Moon Image', moon_image)
cv2.imshow('Filtered Moon Image', filtered_moon_image)

cv2.imwrite("colorImage/filteredColor2.png", filtered_moon_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
