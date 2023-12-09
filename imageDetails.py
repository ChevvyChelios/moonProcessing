import cv2
import matplotlib.pyplot as plt

# Read an image
image_path = "colorImage/fillturedColor.png"
image = cv2.imread(image_path)

# Convert the image from BGR to RGB (Matplotlib uses RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(image_rgb)
plt.title('Image')
plt.axis('off')  # Turn off axis labels
plt.show()
