from re import M
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Dense, Input, Dropout, GlobalAveragePooling2D, MaxPooling2D, Flatten
from keras.regularizers import l2

import os, cv2
import numpy as np
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
img_dir = r'E:\DATA\GEO-GUESS\aug-256\images'
coord_path = r'E:\DATA\GEO-GUESS\aug-256\coordinates\coordinates_aug.txt'
DATA_PATH = r'E:\DATA\GEO-GUESS\aug-256\compressed\256-M.npz'

val_split = 0.1
TARGET_SIZE = (128, 128)
INPUT_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
BATCH_SIZE = 32
EPOCHS = 4
SHUFFLE_BUFFER_SIZE = 500


def load_data():
    with np.load(DATA_PATH) as data:
        xdata, ydata = data['xdata'],data['ydata']
    full_dataset = tf.data.Dataset.from_tensor_slices((xdata, ydata))

    full_dataset = full_dataset.shuffle(1000)
    
    train_dataset = full_dataset.take(4500)
    test_dataset = full_dataset.skip(4500)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    
    return train_dataset, test_dataset


def get_model():

    model = Sequential()

    model.add(Conv2D(128,(3,3), activation='relu',kernel_initializer='he_uniform',padding='same',
    kernel_regularizer=l2(0.001),input_shape=INPUT_SHAPE))

    model.add(Conv2D(128,(3,3), activation='relu',kernel_initializer='he_uniform',padding='same',
    kernel_regularizer=l2(0.001)))

    model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.1))

    model.add(Conv2D(128,(3,3), activation='relu',kernel_initializer='he_uniform',padding='same',
    kernel_regularizer=l2(0.001),input_shape=INPUT_SHAPE))

    model.add(Conv2D(128,(3,3), activation='relu',kernel_initializer='he_uniform',padding='same',
    kernel_regularizer=l2(0.001)))

    model.add(Flatten())
    model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
    #model.add(Dropout(0.1))

    model.add(Dense(16,activation='relu',kernel_initializer='he_uniform'))

    model.add(Dense(2, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mean_squared_error'])

    return model

train_dataset, test_dataset = load_data()

model = get_model()

model.fit(train_dataset, validation_data = test_dataset, epochs=EPOCHS)

model.save(r'E:\models\geo-guess-model.h5')