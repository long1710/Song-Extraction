import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models

## TODO: refactor this before professor meeting tomorrow
def build(config, x_in):
    config['CNN'][]

#####################################3##########################
#(1): model from Convolutional Neural Network based Audio
# Event Classification
def model1_1():
    model = models.Sequential([
        layers.Input(shape =input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(32, (5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels)
    ])
    return model

def model1_2():
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation='relu'),
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (4, 4), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels)
    ])
    return model

def model1_3():
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(5, 5), activation='relu'),
        layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, (4, 4), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels)
    ])
    return model

########################################################################################
#(2): https://arxiv.org/pdf/1811.06669.pdf
def model2_1():
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu',strides=(1,1)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu',strides=(1,1)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu',strides=(1,1)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu',strides=(1,1)),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu',strides=(1,1)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(256, kernel_size=(3,3), activation='relu',strides=(1,1)),
        layers.Conv2D(256, kernel_size=(3,3), activation='relu',strides=(1,1)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(512, kernel_size=(3,3), activation='relu',strides=(1,1)),
        layers.Conv2D(512, kernel_size=(3,3), activation='relu',strides=(1,1)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(50, kernel_size=(1,1), activation='relu',strides=(1,1)),
        layers.AveragePooling2D(pool_size=(2,4), strides=(1,1)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels)
    ])
    return model

#######################################################################
#(3):https://github.com/jordipons/elmarc#:~:text=
#Randomly%20weighted%20CNNs%20for%20(music)%20audio
#%20classification%20This,with%20the%20results%20of%20
#their%20end-to-end%20trained%20counterparts.

