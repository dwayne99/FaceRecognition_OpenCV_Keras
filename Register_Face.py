# Necessary Dependancies
import cv2
import numpy as np
import os

#Load the HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Count function
def get_max(l):
    number = []
    for word in l:
        temp = ''
        for letter in word:
            if letter != '.':
                temp += letter
            else:
                break
        number.append(int(temp))

    return max(number)




# Load functions 
def face_extractor(img):
    # Function detects faces and returns the cropped face 
    # If no face is detected, it returns the input image
    faces = face_classifier.detectMultiScale(img,1.3,5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y+h+50, x:x+w+50]
    
    return cropped_face

# Initialize the webcam
name = input('Enter your name:')
name_path = 'images/' + name

if not os.path.isdir(name_path):
    os.makedirs(name_path)

cap = cv2.VideoCapture(0)


if os.listdir(name_path):
    count = get_max(os.listdir(name_path))
else:
    count = 0

stop = count + 100

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (400, 400))
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = name_path + '/'+ str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == stop: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")

