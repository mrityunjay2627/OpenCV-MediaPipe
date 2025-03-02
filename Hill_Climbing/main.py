import mediapipe as mp
import cv2
import pydirectinput

mp_drawing = mp.solutions.drawing_utils # To draw features on the video frames
mp_holistic = mp.solutions.holistic # To detect and track the drawings in the video
mp_pose = mp.solutions.pose # To detect pose

cap = cv2.VideoCapture(0)

cap.set(3,560) # Width
cap.set(4,400) # Height

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    # confidence means how reliable features selection is. But to avoid overfitting, don't keep confidence 1 (100%)
    while cap.isOpened():
        ret, img = cap.read()
        img = cv2.flip(img, 1) # In camera, our right is left and vice versa. So, to keep our right "right" and left "left", we are flipping the image (frames)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) # frame
        height, width, _ = img.shape

        try:
            right_hand = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * height) # Works for opposite (left)
            left_hand = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height) # Works for opposite (right)
            
            y_mid = height//2 # To accelerate/ deaccelerate (If above y_mid: accelerate ; else: deaccelerate)
            pose = 'move'
            if right_hand[1]<=y_mid:
                pose="acc"
                pydirectinput.keyDown('right') # Press 'right' arrow key for acceleration
                pydirectinput.keyUp('left') # Unpress 'left' arrow key
            elif right_hand[1]>y_mid:
                pose="deacc"
                pydirectinput.keyDown('left')
                pydirectinput.keyUp('right')

            cv2.putText(img, pose, (20,8), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255,255,0), 2)
            cv2.line(img, (0,y_mid), (width, y_mid), (255,0,255), 2)

            if cv2.waitKey(10) & 0xff==ord('q'):
                break
        except:
            pass


cap.release()
cv2.destroyAllWindows()