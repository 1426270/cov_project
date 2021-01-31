import numpy as np
import cv2
import dlib
import cvlib

frontalFaceCascadePath = "/home/butti/PycharmProjects/fh_cov_code/faceDetection_ex17/haarcascade_frontalface_alt.xml"
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
    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)

    detector = dlib.get_frontal_face_detector()
    dets = detector(gray1, 1)

    faces_detected, confidence = cvlib.detect_face(img1)

    img_haar = img1.copy()
    img_hog = img1.copy()
    img_cvlib = img1.copy()

    for (x, y, w, h) in faces1:
        img_haar = cv2.rectangle(img_haar, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for det in dets:
        img_hog = cv2.rectangle(img_hog, (det.tl_corner().x, det.tl_corner().y), (det.br_corner().x, det.br_corner().y), (255, 0, 0), 2)

    for (x, y, x2, y2) in faces_detected:
        img_cvlib = cv2.rectangle(img_cvlib, (x, y), (x2, y2), (255, 0, 0), 2)

    cv2.putText(img_haar, 'HaarCascade', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_hog, 'HOG', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_cvlib, 'Caffe', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    combined = np.hstack((img_haar, img_hog, img_cvlib))

    cv2.imshow('img with detections from video', combined)
    keyboard = cv2.waitKey(delayInMS)
    if keyboard == 'q' or keyboard == 27:
        exit(-1)