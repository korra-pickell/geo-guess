from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Dense, Input, Dropout, GlobalAveragePooling2D, MaxPooling2D, Flatten
from keras.regularizers import l2

import time

data_dir = r'E:\DATA\GEO-GUESS\states\IM100'

val_split = 0.1
TARGET_SIZE = (256, 256)
INPUT_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
BATCH_SIZE = 100
EPOCHS = 50

# Image Loading Needs Pillow Library
# Define the parameters of the Generator
datagen = ImageDataGenerator(rescale=1.0/255,
                             validation_split=val_split,
                             vertical_flip=False,
                             horizontal_flip=True,
                             rotation_range=0)

# ImageDataGenerator one hot encodes. Model needs 'categorical_crossentropy' loss and softmax in output layer

# Define training image generator
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    seed=12,
    subset='training'
)

# Define validation image generator
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    shuffle=True,
    seed=12,
    subset='validation'
)

class_count = len(train_gen.class_indices)


def get_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001), input_shape=INPUT_SHAPE))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(class_count, activation='sigmoid'))
    # compile model
    # opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_model_two():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(class_count, activation='sigmoid'))

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) + softmax for multi
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # + sigmoid for binary

    return model


model = get_model_two()

history = model.fit_generator(train_gen, validation_data=val_gen, epochs=EPOCHS, shuffle=True)
