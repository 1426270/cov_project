import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils.object_detection import non_max_suppression
from imutils import paths

image = cv2.imread("/home/butti/PycharmProjects/cov_project/delete.jpeg")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(rects, weights) = hog.detectMultiScale(image,scale=1.05)

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

for (xA, yA, xB, yB) in pick:
	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

plt.imshow(image)
plt.show()