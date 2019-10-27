# Testing CMAP CNN model

# Data handlers
import numpy as np
import pandas as pd
from glob import glob
import cv2

# Model generator
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.utils import to_categorical

# Load dataset
train = pd.read_csv("/home/nishant/Desktop/IDA Project/mod_data/train.csv")

# Load image paths
img_paths = glob("/home/nishant/Desktop/IDA Project/cmaps/*.0")


# Preparing training and validation data
# Select 2000 images of each type
root = '/home/nishant/Desktop/IDA Project/cmaps/'

keys_0 = train.loc[train.default_ind == 0, 'application_key'].values[:2000]
targs_0 = np.zeros((2000, 1))
imgs_0 = []

keys_1 = train.loc[train.default_ind == 1, 'application_key'].values[:2000]
targs_1 = np.ones((2000, 1))
imgs_1 = []

for i in range(len(keys_0)):
    print("Now processing %d of %d" % (i, len(keys_0)))
    path_0, path_1 = root + str(keys_0[i]), root + str(keys_1[i])
    img_0, img_1 = cv2.imread(path_0), cv2.imread(path_1)
    img_0, img_1 = cv2.resize(img_0, (39, 25)), cv2.resize(img_1, (39, 25))
    imgs_0.append(img_0)
    imgs_1.append(img_1)


# Splitting 75-25 into training and evaluation
X_train = np.array(imgs_0[:1500] + imgs_1[:1500], dtype=np.float32)
X_val = np.array(imgs_0[1500:] + imgs_1[1500:], dtype=np.float32)
y_train = np.array(1500*[0] + 1500*[1], dtype=np.float32)
y_val = np.array(500*[0] + 500*[1], dtype=np.float32)


# Define CNN =====================================================================
# Some model parameters
batch_size = 50
num_classes = 2
epochs = 10

# Input image dimensions
im_rows, im_cols = 39, 25

# Reshape all vectors to tensors
X_train = X_train.reshape((3000, 39, 25, 3))
X_val = X_val.reshape((1000, 39, 25, 3))

# Confirm reshape
print("X_train shape: ", X_train.shape)
print("Train sample: ", X_train[0])
print("Validation sample: ", X_val[0])

# One-hot encode output
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# Build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(im_rows, im_cols, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

# Train model on data
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))

# Scoring
score = model.evaluate(X_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])