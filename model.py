## Import global packages
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import pickle
from tqdm import *
import random
from pathlib import Path
import json
import os
from sklearn.utils import shuffle
import math

## Import Keras necessity's
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers import Input, ELU
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard

## Global configuration
data_path = './data/6/'
throttle_threshold = 0.02 # Data beneath this threshold will be removed

## Load steering angles and images
csv_path = data_path + 'driving_log.csv' # path to csv data
steering_angles = []
images = []

with open(csv_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if float(row[4]) > throttle_threshold:
            steering_angles.append(row[3]) # append fourth column (angle) to steering_angles
            images.append(row[0]) # append first column (image) to images
    steering_angles = np.array(steering_angles, dtype=np.float32) # convert steering_angles to numpy array

print("No. angles = {}".format(len(steering_angles)))
print("No. images = {}".format(len(images)))
print()
print("ANGLE INFORMATION:")
print("Max angle = {:.3f}".format(np.max(steering_angles)))
print("Min angle = {:.3f}".format(np.min(steering_angles)))
print("Mean angle = {:.3f}".format(np.mean(steering_angles)))
print()
print("DONE: Loading angles and image locations")

## Load and normalize the images
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

## Flip image along vertical axis
def flip_image(image, label):
    return np.fliplr(image), -label

## Convert image to HSV and adjust the brightness channel
def adjusted_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    random_bright = .25+np.random.uniform()
    # slice to select brightness channel only
    image[:,:,2] = image[:,:,2]*random_bright
    return image

## Jitter the image and add or subtract a steering angle
def shift_angles(image, steering_angle, trans_range):
    rows, cols, channels = image.shape

    ## Horizontal translation
    tr_x = trans_range*np.random.uniform() - trans_range / 2
    steering_angle = steering_angle + tr_x / trans_range * .4

    trans_matrix = np.float32([[1,0,tr_x],[0,1,0]])
    trans_image = cv2.warpAffine(image,trans_matrix,(cols,rows))
    return trans_image, steering_angle

## Crop the image (to get rid of hood and sky)
def crop_image(image, y1, y2, x1, x2):
    return image[y1:y2, x1:x2]

## Define the X_train and y_train
X_train = images
y_train = steering_angles

## Apply all transformations to an image
def processImage(image, angle, train = True):
    x = cv2.imread(image) # Read the image
    y = angle # Store attached angle
    if train: # If train is set to True
        x,y = shift_angles(x, y, 50) # jitter image
        x = crop_image(x, math.floor(x.shape[0] / 5), x.shape[0] - 20, 50, x.shape[1] - 50) # crop the image (loose sky and hood)
        if np.random.uniform() > 0.5: # Give it a 50% change
            x,y = flip_image(x,y) # to flip the image and angle
        x = adjusted_brightness(x) # adjust brightness
    else: # If in validation mode
        x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV) # Convert image to HSV (normally done in brightness augmentation)
        x = crop_image(x, math.floor(x.shape[0] / 5), x.shape[0] - 20, 50, x.shape[1] - 50) # crop the image
    x = cv2.resize(x, (200, 66)) # Resize the image to the NVIDIA suggested size
    return x, y

def batch_generator_train(data, batch_size = 128):
    data_images, data_angles = data # split data into images and angles
    while 1:
        images = []
        steering = np.zeros(batch_size)
        for i in range(batch_size):
            random_select = random.randint(0, len(data_images) - 1) # Random select an image from the data set
            camera = np.random.choice(['center', 'left', 'right']) # Randomly select which view to use
            if camera == 'left':
                image = data_images[random_select].replace('center', 'left')
                angle = data_angles[random_select] + .25 # Add .25 steering angle if chosen for left camera
            if camera == 'center':
                image = data_images[random_select]
                angle = data_angles[random_select]
            if camera == 'right':
                image = data_images[random_select].replace('center', 'right')
                angle = data_angles[random_select] - .25 # Subtract .25 steering angle if chosen for right camera
            stop_searching = 0
            while stop_searching == 0:
                x, y = processImage(image, angle, True)
                if abs(y) < .15: # If angle is smaller than .15
                    if np.random.uniform() > 1: # Reprocess the image until a bigger angle is found
                        stop_searching = 1
                else:
                    stop_searching = 1
            images.append(np.reshape(x, (1,66,200,3))) # Append the image
            steering[i] = y
        images, steering = (np.vstack(images), steering) # Make sure images is a numpy array
        images, steering = shuffle(images, steering) # Shuffle the images and angles
        yield images, steering # Yield them

def batch_generator_validate(data, batch_size = 128):
    data_images, data_angles = data
    while 1:
        images = []
        steering = np.zeros(batch_size)
        for i in range(batch_size):
            random_select = random.randint(0, len(data_images) - 1) # Random select an image
            x = data_images[random_select]
            y = data_angles[random_select]
            x, y = processImage(x, y, False) # Only crop, resize and turn in to HSV
            images.append(np.reshape(x, (1,66,200,3)))
            steering[i] = y

        steering = np.array(steering, dtype=np.float32)
        yield np.vstack(images), steering

## Hyper parameters
learning_rate = 0.001
batch_size = 128
epochs = 1
keep_prob = 0.2
regulization_val = 0.01

## Define model

input_shape = (66, 200, 3) # Define the input shape for the network

model = Sequential()

# Normalization layer, normalize data with GPU power
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, name='normalize1'))

# 1st Convolution 5x5 filter, 2x2 stride
model.add(Convolution2D(3, 5, 5, activation='relu', border_mode='valid', subsample=(2,2), init='he_normal', name='conv1', W_regularizer=l2(regulization_val)))
model.add(ELU())
model.add(Dropout(keep_prob, name='dropout1'))

# 2nd Convolution 5x5 filter, 2x2 stride
model.add(Convolution2D(24, 5, 5, activation='relu', border_mode='valid', subsample=(2,2), init='he_normal', name='conv2'))
model.add(ELU())
model.add(Dropout(keep_prob, name='dropout2'))

# 3rd Convolution 5x5 filter, 2x2 stride
model.add(Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2,2), init='he_normal', name='conv3'))
model.add(ELU())
model.add(Dropout(keep_prob, name='dropout3'))

# 4th Convolution 3x3 filter, 0x0 stride
model.add(Convolution2D(48, 3, 3, activation='relu', border_mode='valid', subsample=(1,1), init='he_normal', name='conv4'))
model.add(ELU())
model.add(Dropout(keep_prob, name='dropout4'))

# 5th Convolution 3x3 filter, 0x0 stride
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1,1), init='he_normal', name='conv5'))
model.add(ELU())
model.add(Dropout(keep_prob, name='dropout5'))

# Flatten layer
model.add(Flatten())

# 1st Fully connected layer, size=100
model.add(Dense(100, init='he_normal', name='hidden1'))
model.add(ELU())
model.add(Dropout(keep_prob, name='dropout6'))

# 2nd Fully connected layer, size=50
model.add(Dense(50, init='he_normal', name='hidden2'))
model.add(ELU())
model.add(Dropout(keep_prob, name='dropout7'))

# 3rd Fully connected layer, size=10
model.add(Dense(10, init='he_normal', name='hidden3'))
model.add(ELU())

# Output layer
model.add(Dense(1, name='output'))

# Split data into train and validation data
X_split_train, X_split_val, y_split_train, y_split_val = train_test_split(X_train, y_train, test_size=.2)

# Setup generators
train_generator = batch_generator_train((X_split_train, y_split_train), batch_size)
train_validator = batch_generator_validate((X_split_val, y_split_val), batch_size)

# Apply early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min')

model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

for i in range(1, 11): # Run 10 times 1 EPOCH
    history = model.fit_generator(
        train_generator,
        samples_per_epoch=24064,
        nb_epoch=epochs,
        validation_data=train_validator,
        nb_val_samples=4608,
        callbacks=[early_stop]
    )
    ## Save every EPOCH to a seperate file
    model_file = './results/model-epoch-{}.json'.format(i)
    weights_file = './results/model-epoch-{}.h5'.format(i)
    if Path(model_file).is_file():
        os.remove(model_file)
    json_string = model.to_json()
    with open(model_file,'w' ) as f:
        json.dump(json_string, f)
    if Path(weights_file).is_file():
        os.remove(weights_file)
    model.save_weights(weights_file)
    print("SAVED EPOCH {} TO FILE".format(i))
