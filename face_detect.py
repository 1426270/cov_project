# https://realpython.com/face-recognition-with-python/
# training data: https://github.com/arunponnusamy/gender-detection-keras/releases/download/v0.1/gender_dataset_face.zip
# https://github.com/arunponnusamy/gender-detection-keras/blob/master/detect_gender.py

# https://data-flair.training/blogs/python-project-real-time-human-detection-counting/

# data: http://chenlab.ece.cornell.edu/people/Andy/ImagesOfGroups.html        !!!!!!!!!!!!!!!!

# https://github.com/ageitgey/face_recognition
# https://github.com/davisking/dlib/blob/master/python_examples/face_detector.py

# body classifier: https://github.com/nghorbani/homogenus


# todo: check gender classification for all faces
# todo: try different haarcascade.xml files
# todo: try different classifier models?
# todo: openpose?
# todo: caffe !!! https://gist.github.com/GilLevi/c9e99062283c719c03de

import os
import cv2
import dlib
import cvlib
import numpy as np
import pandas as pd
import face_recognition
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# modes
# 0: open-cv + haar_cascade
# 1: open-cv dnn + caffe
# 2: dlib hog
# 3: dlib cnn
mode = 3
modes = ['openCV - HaarCascade', 'openCV - Caffe', 'dlib - HOG', 'dlib - CNN']

overwrite = False # overwrite paths in metadata (only True if detection algorithms changed) --> False saves processing time!
write = False # gets set by code, don't change to True!

# data paths
path_images = os.path.join(os.getcwd(), "data", "Fam4a_used")
path_output = os.path.join(os.getcwd(), "data", "Fam4a_used_out", f"mode_{mode}")
path_output_wrong = os.path.join(path_output, "wrong_number_of_faces")
path_output_correct = os.path.join(path_output, "correct_number_of_faces")
for p in [path_output_wrong, path_output_correct]:
    if not os.path.exists(p):
        os.makedirs(p)
path_results_face_detect = os.path.join(os.getcwd(), "results", "results_face_detect.csv")
if not os.path.isfile(path_results_face_detect):
    df_results = pd.DataFrame({'modes': modes, 'val': [np.nan]*4})
else:
    df_results = pd.read_csv(path_results_face_detect)

loc_list = []
path_face_locs = os.path.join(path_output, "locs.npy")
if os.path.isfile(path_face_locs):
    loc_list = np.load(path_face_locs, allow_pickle=True)
if overwrite:
    loc_list = []


path_cascade = os.path.join("haar_cascade", "haarcascade_frontalface_alt.xml")
path_keras_model = os.path.join(os.getcwd(), "models", "gender_detection.model")
path_meta = os.path.join(path_images, "Fam4a_used_meta.csv")

# read meta data
df_meta = pd.read_csv(path_meta)

# load pre-trained keras model
model = load_model(path_keras_model)
classes = ['m', 'f']

faces = None
correct_number = 0
for i, row in df_meta.iterrows():
    print(f"Image {i + 1} of {len(df_meta)}")
    # Read the image
    image = cv2.imread(os.path.join(path_images, row["filename"]))
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if df_meta[f'path_{mode}'].isnull().values.any() or overwrite:
        write = True
        # Detect faces in the image
        if mode == 0:
            # create the haar cascade
            faceCascade = cv2.CascadeClassifier(path_cascade)
            faces = faceCascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4,
                                                 minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if mode == 1:
            faces_detected, confidence = cvlib.detect_face(image)
            if len(faces_detected) == 0:
                faces = np.array([])
            else:
                faces_detected = np.array(faces_detected)
                faces = faces_detected.copy()
                faces = faces.clip(min=0)
                faces[:,2] = faces_detected[:,2] - faces_detected[:,0]
                faces[:,3] = faces_detected[:,3] - faces_detected[:,1]
        if mode == 2:
            detector = dlib.get_frontal_face_detector()
            dets = detector(image, 1)
            faces = np.array([[det.tl_corner().x, det.tl_corner().y, det.tr_corner().x - det.tl_corner().x,
                               det.br_corner().y - det.tr_corner().y] for det in dets])
        if mode == 3:
            # uses hog per default --> dlib.get_frontal_face_detector()!!!! cnn --> dlib.cnn_face_detection_model_v1
            faces_detected = np.array(face_recognition.face_locations(grey, model="cnn"))
            faces = np.zeros(faces_detected.shape)
            faces[:,0] = faces_detected[:,3]
            faces[:,1] = faces_detected[:,0]
            faces[:,2] = faces_detected[:,1] - faces_detected[:,3]
            faces[:,3] = faces_detected[:,2] - faces_detected[:,0]

        faces = faces.astype(np.int)
        if len(faces) > 0:
            faces_sorted = faces[faces[:, 0].argsort()]
        loc_list.append(faces_sorted)
        if len(faces_sorted) == row["number_of_faces"]:
            correct_number += 1
            # path_output_img = os.path.join(path_output, "correct_number_of_faces", row["filename"])
            df_meta[f'path_{mode}'][i] = os.path.join(path_output, "correct_number_of_faces", row["filename"])
        else:
            # path_output_img = os.path.join(path_output, "wrong_number_of_faces", row["filename"])
            df_meta[f'path_{mode}'][i] = os.path.join(path_output, "wrong_number_of_faces", row["filename"])
    else:
        # simply load all positions
        faces_sorted = loc_list[i]

    path_output_img = df_meta[f'path_{mode}'][i]

    # Draw a rectangle around the faces and classify
    if len(faces_sorted) > 0:
        title = "|"
        for j, (x, y, w, h) in enumerate(faces_sorted):
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # crop the detected face region
            face_crop = np.copy(image[y:y+h, x:x+w])

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (50, 50))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = model.predict(face_crop)[0]
    
            # get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]
    
            if len(faces_sorted) == row["number_of_faces"]:
                label = " {:.2f}% {} ({}) |".format(conf[idx] * 100, label, row["genders"][j])
            else:
                label = " {:.2f}% {} |".format( conf[idx] * 100, label)
            if j == 3:
                label = label + "\n"
    
            # write label and confidence above face rectangle
            # cv2.rectangle(image, (x, y-30), (x+w, y), (255, 255, 255), -1)
            # Y = y - 10 if y - 10 > 10 else y + 10
            # cv2.putText(image, label, (x, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            title += label

    plt.imshow(image)
    plt.title(title, fontsize=9)
    plt.savefig(path_output_img)
    plt.close()

if write:
    df_meta.to_csv(path_meta)
    np.save(path_face_locs, loc_list, allow_pickle=True)
    df_results['val'][mode] = correct_number
    df_results.to_csv(path_results_face_detect)
    print(correct_number, df_meta.shape)