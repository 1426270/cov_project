# https://realpython.com/face-recognition-with-python/
# training data: https://github.com/arunponnusamy/gender-detection-keras/releases/download/v0.1/gender_dataset_face.zip
# https://github.com/arunponnusamy/gender-detection-keras/blob/master/detect_gender.py

import os
import cv2
import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Data
path_data = os.path.join(os.getcwd(), "data")
path_image = os.path.join(path_data, "abba.png")
path_cascade = os.path.join("haar_cascade", "haarcascade_frontalface_alt.xml")
path_model = os.path.join(os.getcwd(), "models", "gender_detection.model")

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(path_cascade)

# Read the image
image = cv2.imread(path_image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


classes = ['m', 'f']
# load pre-trained model
model = load_model(path_model)

# Detect faces in the image
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # crop the detected face region
    face_crop = np.copy(image[y:y+h, x:x+w])

    # preprocessing for gender detection model
    face_crop = cv2.resize(face_crop, (96, 96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # apply gender detection on face
    conf = model.predict(face_crop)[0]
    print(conf)
    print(classes)

    # get label with max accuracy
    idx = np.argmax(conf)
    label = classes[idx]

    label = "{}: {:.2f}%".format(label, conf[idx] * 100)

    Y = y - 10 if y - 10 > 10 else y + 10

    # write label and confidence above face rectangle
    cv2.putText(image, label, (x, Y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)




plt.imshow(image)
plt.show()
