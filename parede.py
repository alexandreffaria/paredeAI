import cv2
import numpy as np


def show_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(5000)


# Load the image
image_path = 'parede.png'
image = cv2.imread(image_path)
# image = cv2.resize(image, (224, 224))
show_image("Original Image", image)

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
show_image("HSV Image", hsv)

# Define the color range for the holds
lower_color = np.array([16, 15, 84])
upper_color = np.array([107, 194, 247])

# Filter the image based on the color range
mask = cv2.inRange(hsv, lower_color, upper_color)
show_image("Color Mask", mask)

# Apply a Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(mask, (5, 5), 0)
show_image("Blurred Image", blurred)

# Apply morphological transformations
kernel = np.ones((3, 3), np.uint8)
opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
show_image("Opened Image", opened)

closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
show_image("Closed Image", closed)

# Find contours in the binary image
contours, _ = cv2.findContours(
    closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on their size
min_area = 25  # Adjust this value based on the minimum size of the holds
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# Draw detected contours on the original image
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save the detected holds image
cv2.imwrite('detected_holds.png', image)

show_image('Detected Holds', image)
cv2.destroyAllWindows()
