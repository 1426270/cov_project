import os
import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# data paths
path_images = os.path.join(os.getcwd(), "data", "Fam4a")
path_output = os.path.join(os.getcwd(), "data", "train_data_out")

for p in [path_output]:
    if not os.path.exists(p):
        os.makedirs(p)
path_meta = os.path.join(path_output, "train_data.csv")

# read meta data
# df_meta = pd.read_csv(path_meta)


count = 0
for filename in os.listdir(path_images):
    # Read the image
    image = cv2.imread(os.path.join(path_images, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 1)
    faces = np.array([[det.tl_corner().x, det.tl_corner().y, det.br_corner().x, det.br_corner().y] for det in dets])

    plt.imshow(image)
    plt.show()
    for j, (x1, y1, x2, y2) in enumerate(faces):
        count = count + 1
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(image[y1:y2, x1:x2])

        plt.imshow(face_crop)

        plt.show()