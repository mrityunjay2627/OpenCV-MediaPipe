import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import os

current_directory = os.getcwd()
img_path = os.path.join(current_directory, 'written_letter.png')

reader = easyocr.Reader(['en'])
result = reader.readtext(img_path)

# print(result[0][0])
# print(result[0][0][2])

top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])

text = result[0][1]

img = cv2.imread(img_path)



# img = cv2.rectangle(img,top_left,bottom_right,(0,255,255),2)
# img = cv2.putText(img,text,top_left,cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
# plt.imshow(img)
# plt.show()


'''
Below will not work for all text since OCR didn't parse the text from image correctly
'''

count = 0

for oneimage in result:
    top_left = tuple(oneimage[0][0])
    bottom_right = tuple(oneimage[0][2])
    text = oneimage[1]
    img = cv2.rectangle(img,top_left,bottom_right,(0,255,255),2)
    img = cv2.putText(img,text,top_left,cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    if count>4:
        break
    count+=1

plt.imshow(img)
plt.show()