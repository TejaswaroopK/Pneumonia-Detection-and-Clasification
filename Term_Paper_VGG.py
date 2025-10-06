from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import models
from tensorflow.keras.applications import VGG16

class CNN():
    def cnn_vgg(input_shape=(124, 124, 3)):
        model = models.Sequential()
        model.add(VGG16(weights='imagenet', include_top=False,    input_shape=input_shape))
        model.add(Flatten())
        model.add(Dense(1024, activation=('relu'), input_dim = 512))
        model.add(Dense(512, activation=('relu')))
        model.add(Dense(256, activation=('relu')))
        model.add(Dense(128, activation=('relu')))
        model.add(Dense(2, activation='softmax'))
        return model