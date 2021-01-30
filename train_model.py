from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os


def build(width, height, depth):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # model.add(Conv2D(128, (3,3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv2D(128, (3,3), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation("sigmoid"))

    return model


# initial parameters
epochs = 100
lr = 1e-3
batch_size = 32
img_dims = (50, 50, 3)

data = []
labels = []

path_images = os.path.join(os.getcwd(), "data", "train_data_out")
# create ground-truth label from the image path

for gender in ["man", "woman"]:
    for img in os.listdir(os.path.join(path_images, gender)):
        image = cv2.imread(os.path.join(path_images, gender, img))

        image = cv2.resize(image, (img_dims[0], img_dims[1]))
        image = img_to_array(image)
        data.append(image)

        if gender == "woman":
            label = 1
        else:
            label = 0

        labels.append([label])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# augmenting datset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.1,
                         horizontal_flip=True, fill_mode="nearest")

# build model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2])

# compile the model
opt = Adam(lr=lr, decay=lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs, verbose=1)

# save the model to disk
model.save(os.path.join("models", "own_model_own_data.model"))

# plot training/validation loss/accuracy
plt.style.use("seaborn-whitegrid")
df = pd.DataFrame(H.history)

fig1, ax1 = plt.subplots()
df[['loss', 'val_loss']].plot(ax=ax1)
ax1.set_xlabel("Epoch #")
ax1.set_ylabel("Loss")
plt.savefig(os.path.join("models", "loss.png"))

fig2, ax2 = plt.subplots()
df[['accuracy', 'val_accuracy']].plot(ax=ax2)
ax2.set_xlabel("Epoch #")
ax2.set_ylabel("Accuracy")
plt.savefig(os.path.join("models", "accuracy.png"))