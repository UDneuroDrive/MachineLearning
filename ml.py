from __future__ import absolute_import, division, print_function, unicode_literals

from keras.preprocessing.image import img_to_array
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
# from pyimagesearch.smallervggnet import SmallerVGGNet
# import cv2
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Flatten, Dense, Dropout
# from keras.preprocessing.image import ImageDataGenerator
# import time



import matplotlib
import matplotlib.pyplot as plt
import pandas
from imutils import paths
import numpy as np
import cv2




from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import time
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


IMG_SIZE = (96,96,3)# Replace with the size of your images
NB_CHANNELS = 3# 3 for RGB images or 1 for grayscale images
BATCH_SIZE = 32# Typical values are 8, 16 or 32
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
EPOCHS=10


labels = []

data =[]

imagePaths = sorted(list(paths.list_images("second_data/images")))
try:
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
except:
    print("err")	

data = np.array(data, dtype="float") / 255.0



labels = np.array(pandas.read_csv("second_data/data.csv"))
labels = np.array(labels[:data.shape[0],1])

classNames = ['l','r','c']
labelencoder_y_1 = LabelEncoder()

encoded_labels = to_categorical(labelencoder_y_1.fit_transform(labels))

print(encoded_labels.shape)
print(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
	encoded_labels, test_size=0.2, random_state=42)

NB_TRAIN_IMG = trainX.shape[0]# Replace with the total number training images
NB_VALID_IMG = testX.shape[0]# Replace with the total number validation images
print(NB_TRAIN_IMG)
print(NB_VALID_IMG)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


model = Sequential([
    Conv2D(filters=32, 
               kernel_size=(2,2), 
               strides=(1,1),
               padding='same',
               input_shape=(IMAGE_DIMS[1],IMAGE_DIMS[0],IMAGE_DIMS[2]),
               data_format='channels_last'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2),
                     strides=2),
    Conv2D(filters=64,
               kernel_size=(2,2),
               strides=(1,1),
               padding='valid'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2),
                     strides=2),
    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.25),
    Dense(3),
    Activation('sigmoid')
])



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

test_loss, test_acc = model.evaluate(testX,  testY, verbose=2)

print('\nTest accuracy:', test_acc)

prediction_single = model.predict(testX)

for i in range(testX.shape[0]):
    
    print("predicted {} actual {}", np.argmax(prediction_single[i]),testY[i])


def getPrediction(testData, index):
    predictions = model.predict(testData)
    print(np.argmax(predictions[index]))
