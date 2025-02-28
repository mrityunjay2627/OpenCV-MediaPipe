import mediapipe as mp
import cv2

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ref, frame = cap.read()
#     cv2.imshow('Web came',frame)
    
#     if cv2.waitKey(10) & 0xff==ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


mp_drawing = mp.solutions.drawing_utils # To draw features on the video frames
mp_holistic = mp.solutions.holistic # To detect and track the drawings in the video

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    # confidence means how reliable features selection is. But to avoid overfitting, don't keep confidence 1 (100%)
    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        img = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) # frame
        mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('Holistic', img)

        if cv2.waitKey(10) & 0xff==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()