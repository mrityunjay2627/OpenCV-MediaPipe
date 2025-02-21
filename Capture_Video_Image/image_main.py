import cv2
import os

# Get the current working directory
current_directory = os.getcwd()

# Define thefile name
image = "wallpaper.jpg"  # Change this to your desired file name

# Join the directory
image_path = os.path.join(current_directory, image)

# Image View
img = cv2.imread(image_path)
print(img.shape)
# cv2.imshow('Image',img)

# Resize
img = cv2.resize(img,(200,300))
print(img.shape)
# cv2.imshow('Resized Image',img)

# Grayscale (0-255 shades)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # RGB is written as BGR in OpenCV
print(gray.shape)
# cv2.imshow('Grayscale Image',gray)

# HSV (Change the channels from 3 to 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsv.shape) # Although channels = 3, they work as channels = 1
cv2.imshow('HSV Image',hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
