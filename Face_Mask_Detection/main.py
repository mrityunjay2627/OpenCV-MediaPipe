import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import os

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(
    rescale = 1/255,
    shear_range = 0.2, # Magnitude of distortion
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(
    rescale = 1/255
)

current_directory = os.getcwd()

train_data = "data/train/"  
test_data = "data/test/"

train_data_path = os.path.join(current_directory, train_data)
test_data_path = os.path.join(current_directory, test_data)

train_set = train_datagen.flow_from_directory(
    train_data_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)

test_set = train_datagen.flow_from_directory(
    test_data_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary'
)

# history = model.fit(
#     train_set,
#     epochs = 10,
#     validation_data = test_set
# )

# model.save('mask_detector.h5', history)

mymodel = load_model('mask_detector.h5')

haar = "haarcascade_frontalface_default.xml"

# Join the directory
haarpath = os.path.join(current_directory, haar)

# Cascade (to detect important features in an image)
face_cascade = cv2.CascadeClassifier(haarpath) # Front face properties will be detected

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200) # set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # set height

while cap.isOpened():
    _, img = cap.read()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # To read features, we need edges and for edges, we convert image to grayscale
    faces = face_cascade.detectMultiScale(img, 1.1, 4) # Nearest neighbors features to be taken into consideration
    
    for (x,y,w,h) in faces:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg', face_img) # Save image that is being detected/predicted

        test_img = image.load_img('temp.jpg', target_size=(224,224,3))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)

        pred = mymodel.predict(test_img)[0][0]

        if pred==1: # No Mask
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) # face
            cv2.rectangle(img, (x,y-40), (x+w,y), (0,255,0), -1) # label box (mask/no mask) that is above person's face
            cv2.putText(img, 'No Mask', (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 2)

        else:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) # face
            cv2.rectangle(img, (x,y-40), (x+w,y), (0,255,0), -1) # label box (mask/no mask) that is above person's face
            cv2.putText(img, 'Mask', (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('Image', img)

    if cv2.waitKey(10) & 0xff==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


