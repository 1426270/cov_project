import numpy as np
import cv2

frontalFaceCascadePath = "/home/butti/PycharmProjects/fh_cov_code/faceDetection_ex17/haarcascade_frontalface_alt.xml"

# 1. Load the cascades
face_cascade = cv2.CascadeClassifier(frontalFaceCascadePath)

delayInMS = 100


capture = cv2.VideoCapture(0)
if not capture.isOpened:
    print('Unable to open: ' + str(0))
    exit(0)
while True:
    print('processing frame')
    ret, frame = capture.read()
    if frame is None:
        print('no frame loaded')
        break

    img1 = frame.copy()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray1, 1.3, 5)
    for (x, y, w, h) in faces:
        img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('img with detections from video', img1)
        keyboard = cv2.waitKey(delayInMS)
        if keyboard == 'q' or keyboard == 27:
            exit(-1)