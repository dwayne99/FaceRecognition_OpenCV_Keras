# Face Recognition

# necessary packages
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json 
import random 
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os

# For naming the classes
l_ = []
for f in os.listdir('data/train/'):
    l_.append(f.upper())

l_ = sorted(l_)
people = {}
for i,person in enumerate(l_):
    people[i] = person.title()


# people = {
#     0:'Abigail',
#     1: 'Dwayne',
#     2: 'Papa',
#     3: 'Ronnie',
#     4: 'Veronica'
# }


# Loading the trained model 
model = load_model('vgg_model1.h5')


# Loading the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        pid = np.argmax(pred,axis=1)[0]
        name="None matching"
        name = people[pid]
            
        cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()




