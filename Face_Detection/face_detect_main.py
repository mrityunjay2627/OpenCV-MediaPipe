import numpy as np
import cv2
import os

current_directory = os.getcwd()

haar = "haarcascade_frontalface_default.xml"

# Join the directory
haarpath = os.path.join(current_directory, haar)

# Cascade (to detect important features in an image)
face_cascade = cv2.CascadeClassifier(haarpath) # Front face properties will be detected

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # To read features, we need edges and for edges, we convert image to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) # Nearest neighbors features to be taken into consideration

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), color=(0,0,255), thickness=2)
        text = cv2.putText(img, "Face", (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 , color=(0,0,255) , thickness=2)
    cv2.imshow("Face", img)
    if cv2.waitKey(27) & 0xff==ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break