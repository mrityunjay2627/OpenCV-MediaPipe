import numpy as np
import cv2
import matplotlib.pyplot as plt

webcam = cv2.VideoCapture(0) # 0 means our webcam feed

while True:
    _, img = webcam.read()

    # Converting image to HSV (Hue, Saturation, Value) to increase sharpness which will help in object detection (better than BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # detect color in image
    red_L = np.array([135,85,115],np.uint8) # Lower range of red color in (0-255) range
    red_U = np.array([185,250,250],np.uint8) # Upper range of red color in (0-255) range

    # Masking (Detect important unknown figures)
    redM = cv2.inRange(hsv, red_L, red_U)

    # Pass this kernel on image to detect the required color figures from the image
    kernel = np.ones((5,5), 'uint8')
    
    # Dilate
    redM = cv2.dilate(redM, kernel) # To add/get clear properties of image as kernel iterates to different parts of image

    # Bitwise
    '''
    The reason `img` is sent twice in the function `cv2.bitwise_and(img, img, mask=redM)` is because it's performing a **bitwise AND** operation on the image with itself.

    In a bitwise AND operation, the function compares the pixel values of the two images. Each pixel in the first image is compared with the corresponding pixel in the second image. The result is a new image where each pixel's value is the result of performing the AND operation on the values of the corresponding pixels from both images.

    Here's a simple breakdown:
    - The first `img` is the source image you want to apply the operation on.
    - The second `img` is essentially the "second operand" in the operation, which is also your original image in this case. This means you're comparing each pixel of `img` with itself.

    Now, the `mask=redM` part limits the operation to only the areas where the mask `redM` is non-zero. So, in areas where `redM` has value (i.e., is not zero), the bitwise AND is applied. In areas where `redM` is zero, the result will be zero, meaning those areas of the image will be "cut out" or "masked."

    In short:
    - The function is doing a **bitwise AND** between the image and itself, but only in the areas where `redM` allows (masking).

    '''
    red = cv2.bitwise_and(img, img, mask=redM) # Compare image pixels with mask pixels and get the required color figures

    # Contour (Track the color in the image)
    cont, hier = cv2.findContours(redM, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # RETR_TREE measures the intensity of the color, and CHAIN_APPROX_SIMPLE defines a boundary around the required color figure(s) in the image

    for pix, cont in enumerate(cont):
        area = cv2.contourArea(cont)
        if area>300:
            x, y, w, h = cv2.boundingRect(cont)
            img = cv2.rectangle(img, (x,y), (x+w,y+h), color=(0,0,255), thickness=2)
            text = cv2.putText(img, "Red Color", (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 , color=(0,0,255) , thickness=2)
    cv2.imshow("Red Color Detection", img)
    if cv2.waitKey(27) & 0xff==ord('q'):
        cv2.destroyAllWindows()
        break