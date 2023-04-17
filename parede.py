import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Load the pre-trained model
model = MobileNetV2(weights='imagenet', include_top=False)

# Load the image
image_path = 'parede.png'
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))

# Preprocess the image
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)

# Perform object detection
predictions = model.predict(image)

# Define a threshold for hold detection
threshold = 0.5

# Extract detected holds
holds = []

for i in range(predictions.shape[1]):
    for j in range(predictions.shape[2]):
        if predictions[0, i, j, 0] > threshold:
            holds.append((i, j))

# Convert the image back to the original format
display_image = ((image[0] + 1) * 127.5).astype(np.uint8)

# Visualize the detected holds
for hold in holds:
    x, y = hold
    x *= 8  # Scale back to the original image size
    y *= 8  # Scale back to the original image size
    cv2.circle(display_image, (x, y), 5, (0, 255, 0), 2)

cv2.imshow('Detected Holds', display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
