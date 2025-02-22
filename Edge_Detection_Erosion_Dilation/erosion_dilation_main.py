'''
Erosion helps in removing noise.
Dilation means to add/get clear properties.
'''

import numpy as np
import cv2, os

current_directory = os.getcwd()

# Image
image = "wallpaper.jpg"  # Change this to your desired file name

image_path = os.path.join(current_directory, image)

img = cv2.imread(image_path)
img = cv2.resize(img,(840,660))

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny_img = cv2.Canny(gray_img,150,200)

# Erosion
kernel = np.ones((1,1),np.uint8) # Dimensions are hyperparameters
erode_img = cv2.erode(canny_img, kernel, iterations=1) # Removes noise as kernel iterates to different parts of canny image
# cv2.imshow('Eroded Image',erode_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Dilation
kernel = np.ones((4,4),np.uint8) # Dimensions are hyperparameters
dil_img = cv2.dilate(canny_img,kernel=kernel,iterations=1) # To add/get clear properties of image as kernel iterates to different parts of canny image
cv2.imshow('Dilated Image',dil_img)
cv2.waitKey(0)
cv2.destroyAllWindows()