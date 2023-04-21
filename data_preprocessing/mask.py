import cv2
import numpy as np

# Load the image
image = cv2.imread("../image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a binary mask based on a threshold value
ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Show the binary mask and the masked image
cv2.imshow("Binary Mask", mask)
masked_image = cv2.bitwise_and(image, image, mask=mask)
# cv2.imshow("Masked Image", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()