# https://realpython.com/face-recognition-with-python/

import os
import cv2
import sys
import matplotlib.pyplot as plt

# Data
path_data = os.path.join(os.getcwd(), "data")
imagePath = os.path.join(path_data, "abba.png")
cascPath = os.path.join(path_data, "haarcascade_frontalface_default.xml")

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(image)
plt.show()
