import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils # To draw features on the video frames
mp_holistic = mp.solutions.holistic # To detect and track the drawings in the video
mp_pose = mp.solutions.pose # To detect pose

cap = cv2.VideoCapture(0)

counter, stage = 0, 0

def calculate_angle(a,b,c):
    a = np.array(a) # Shoulder
    b = np.array(b) # Elbow
    c = np.array(c) # Wrist

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if angle > 180:
        angle = 360.0 - angle
    
    return angle

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = pose.process(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) # frame

        # Extracting Landmarks
        landmarks = results.pose_landmarks.landmark

        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # print(angle)
        cv2.putText(img, str(angle), tuple(np.multiply(elbow, [640,480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),1)

        if angle > 160:
            stage = 'down'
        if angle < 30 and stage == 'down':
            stage = 'up'
            counter += 1
            # print(counter)

        cv2.rectangle(img, (0,0), (225,73), (245,117,16), -1)

        cv2.putText(img, 'CNT', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(img, str(stage), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)



        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(180,105,255),thickness=2,circle_radius=3),
                                 mp_drawing.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=3))
         
        cv2.imshow("Gym Assistant", img)

        if cv2.waitKey(10) & 0xff==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

