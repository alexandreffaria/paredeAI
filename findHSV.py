import cv2
import numpy as np


def nothing(x):
    pass


def update_mask():
    lower_color = np.array([cv2.getTrackbarPos("Hue Min", "Trackbars"),
                            cv2.getTrackbarPos("Sat Min", "Trackbars"),
                            cv2.getTrackbarPos("Val Min", "Trackbars")])

    upper_color = np.array([cv2.getTrackbarPos("Hue Max", "Trackbars"),
                            cv2.getTrackbarPos("Sat Max", "Trackbars"),
                            cv2.getTrackbarPos("Val Max", "Trackbars")])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    cv2.imshow("Color Mask", mask)


# Load the image
image_path = 'parede.png'
image = cv2.imread(image_path)
# image = cv2.resize(image, (224, 224))

# Convert the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create trackbars for adjusting the HSV color range
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Val Min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Val Max", "Trackbars", 255, 255, nothing)

while True:
    update_mask()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
