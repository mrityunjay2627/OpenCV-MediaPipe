import cv2
import os

current_directory = os.getcwd()

# Image
image = "wallpaper.jpg"  # Change this to your desired file name

image_path = os.path.join(current_directory, image)

img = cv2.imread(image_path)
img = cv2.resize(img,(840,660))

# Video
video = "videoplayback.mp4"  # Change this to your desired file name

video_path = os.path.join(current_directory, video)

vid = cv2.VideoCapture(video_path)



''' Edge Detection (Works only on grayscale images)'''

# Image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Canny Image (Detects Edges)
canny_img = cv2.Canny(gray_img,150,200) # 150,200 is filter/kernel size

cv2.imshow('Canny Image',canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Video
# while True:
#     try:
#         res, frame = vid.read()
#         gray_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Canny Video (Detects Edges)
#         canny_vid = cv2.Canny(gray_vid,150,200) # 150,200 is filter/kernel size

#         cv2.imshow('video',canny_vid)
#     except:
#         pass

#     if cv2.waitKey(1) & 0xff==ord('q'):
#         break

# cv2.destroyAllWindows()
