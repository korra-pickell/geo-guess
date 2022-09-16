from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from keras import applications
from keras import layers, models
import time


data_dir = r'E:\DATA\GEO-GUESS\states\IM100'

val_split = 0.1
TARGET_SIZE = (256,256)
INPUT_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
BATCH_SIZE = 16
EPOCHS = 50

# Image Loading Needs Pillow Library
# Define the parameters of the Generator
datagen = ImageDataGenerator(rescale=1.0/255,
                             validation_split=val_split,
                             vertical_flip=False,
                             horizontal_flip=True,
                             zoom_range=0.15,
                             shear_range=0.15,
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

    base_model = applications.InceptionResNetV2(weights='imagenet',
                                                include_top=False,
                                                input_shape=INPUT_SHAPE)
    base_model.trainable = True
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.6))
    model.add(Dense(class_count, activation='sigmoid'))

    #model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model


def get_model_vgg():

    base_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = True
    
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(50, activation='relu')
    dense_layer_2 = layers.Dense(20, activation='relu')
    prediction_layer = layers.Dense(class_count, activation='sigmoid')


    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model


model = get_model_vgg()
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, shuffle=True)