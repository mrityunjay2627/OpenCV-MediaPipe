import cv2
import os

# Get the current working directory
current_directory = os.getcwd()

# Define thefile name
video = "videoplayback.mp4"  # Change this to your desired file name

# Join the directory
video_path = os.path.join(current_directory, video)

vid = cv2.VideoCapture(video_path)

while True:
    try:
        res, frame = vid.read()
        cv2.imshow('video',frame)
    except:
        pass

    if cv2.waitKey(1) & 0xff==ord('q'):
        break

cv2.destroyAllWindows()