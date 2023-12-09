import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ObjectSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Segmentation App")
        self.image_path = None
        self.original_image = None
        self.segmented_image = None

        # Create frames
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.create_widgets()

    def create_widgets(self):
        # Load Image Button
        load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image)
        load_button.pack(pady=10)

        # Object Segmentation Button
        segment_button = tk.Button(self.button_frame, text="Segment Objects", command=self.segment_objects)
        segment_button.pack(pady=10)

        # Image Display
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

    def load_image(self):
        self.image_path = filedialog.askopenfilename()

        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def segment_objects(self):
        if self.original_image is not None:
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to segment objects
            _, thresholded = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the original image
            self.segmented_image = self.original_image.copy()
            cv2.drawContours(self.segmented_image, contours, -1, (0, 255, 0), 2)

            # Display the segmented image
            self.display_image(self.segmented_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectSegmentationApp(root)
    root.mainloop()
