import cv2
import numpy as np

# Load the image
image_path = 'parede.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(5000)

# Apply a Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow('Blurred Image', blurred)
cv2.waitKey(5000)

# Use adaptive thresholding to create a binary image
binary = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
cv2.imshow('Binary Image', binary)
cv2.waitKey(5000)

# Find contours in the binary image
contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on their size
min_area = 10  # Adjust this value based on the minimum size of the holds
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

# Draw detected contours on the original image
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Detected Holds', image)
cv2.waitKey(5000)
cv2.destroyAllWindows()
