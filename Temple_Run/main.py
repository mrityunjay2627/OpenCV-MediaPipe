import mediapipe as mp
import cv2
import  pydirectinput

mp_drawing = mp.solutions.drawing_utils # To draw features on the video frames
mp_holistic = mp.solutions.holistic # To detect and track the drawings in the video
mp_pose = mp.solutions.pose # To detect pose

cap = cv2.VideoCapture(0)

cap.set(3,720) # Width
cap.set(4,540) # Height

pose = ''
status = 0 # Game status (pausedc (0)/running (1))

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    # confidence means how reliable features selection is. But to avoid overfitting, don't keep confidence 1 (100%)
    while cap.isOpened():
        success, img = cap.read()
        img = cv2.flip(img, 1) # In camera, our right is left and vice versa. So, to keep our right "right" and left "left", we are flipping the image (frames)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) # frame
        height, width, _ = img.shape

        try:
            left_hand = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width,
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height) # Works for opposite (left) due to image flip.
            
            line_x1 = width//3
            line_x2 = 2*(width//3)
            line_y1 = height//3
            line_y2 = 2*(height//3)

            if left_hand[0]>line_x2 and left_hand[1]>line_y1 and status == 0:
                pose = 'Start'
                pydirectinput.keyDown('space')
                pydirectinput.keyUp('space')
                status=1
            elif left_hand[0]>line_x2 and left_hand[1]>line_y1 and left_hand[1]<line_y2 and status==1:
                pose = 'Right'
                pydirectinput.keyDown('right')
                pydirectinput.keyUp('right')
            elif left_hand[0]<line_x1 and left_hand[1]>line_y1 and left_hand[1]<line_y2 and status==1:
                pose = 'Left'
                pydirectinput.keyDown('left')
                pydirectinput.keyUp('left')
            elif left_hand[1]<line_y1 and status==1:
                pose = 'Jump'
                pydirectinput.keyDown('up')
                pydirectinput.keyUp('up')
            elif left_hand[1]>line_y2 and status==1:
                pose = 'Slide'
                pydirectinput.keyDown('down')
                pydirectinput.keyUp('down')
            elif status==0:
                pose = 'Please start the game'
            else:
                pose = 'Run'
        except:
            pass

        cv2.putText(img, pose, (10,30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,0), 2)
        cv2.line(img, (width//3,0), (width//3, height), (255,0,255), 2)
        cv2.line(img, ((2*width//3),0), ((2*width//3), height), (255,0,255), 2)
        cv2.line(img, (0,height//3), (width, height//3), (255,0,255), 2)
        cv2.line(img, (0,(2*height//3)), (width, (2*height//3)), (255,0,255), 2)

        # mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        #                          mp_drawing.DrawingSpec(color=(180,105,255),thickness=5,circle_radius=8),
        #                          mp_drawing.DrawingSpec(color=(255,255,255),thickness=10,circle_radius=10))

        cv2.imshow('Temple Run', img)

        if cv2.waitKey(10) & 0xff==ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
