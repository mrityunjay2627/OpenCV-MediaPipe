import cv2
import numpy as np
from matplotlib import pyplot as plt

# Images are stored in numpy arrays
blank = np.zeros((500,500,3), dtype='uint8') # Black

plt.imshow(blank) # Black color

blank[:] = 0,255,0 # Green

plt.imshow(blank) # Green color

blank[:] = 255,0,0 # Red

plt.imshow(blank) # Red color

blank[200:300,300:400] = 0,255,0 # Green

plt.imshow(blank) # Small green square in red square


### Drawing Rectangle ###
rect = cv2.rectangle(blank,(0,0),(250,250), (0,255,0), thickness=2) # Here, blank is numpy array (with color), (0,0) and (255,255) are rectangle dimensions and last parameter is color of the dimension box thickness
plt.imshow(rect)

rect_filled = cv2.rectangle(blank,(0,0),(250,250), (0,255,0), thickness=cv2.FILLED) # Thickness (FILLED) with replace the blank color with color mentioned as parameter
plt.imshow(rect_filled)


### Drawing Circle ###
bz = np.zeros((500,500,3), dtype='uint8')
cir = cv2.circle(bz, (bz[1]//2, bz.shape[0]//2), radius=40, color=(0,0,255), thickness=5) # A blue circle of thickness two, radius 40 and center coordinates bz.shape[1]//2, bz.shape[0]//2 will appear on black (numpy array) background
plt.imshow(cir)


### Drawing Line ###
line = cv2.line(bz, (0,0), (bz[1]//2, bz.shape[0]//2), color=(255,255,255), thickness=10) # A white line of thickness 10 from coordinates (0,0) to coordinates bz.shape[1]//2, bz.shape[0]//2 will appear on black (numpy array) background
plt.imshow(line)


### Text Insertion
bm = np.zeros((500,500,3), dtype='uint8')
text = cv2.putText(bm,'Hello', (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), thickness=10) # A white text (Hello) of thickness 10, starting from (100,100) and of font size 1 and font type HERSHEY_TRIPLEX
plt.imshow(text)