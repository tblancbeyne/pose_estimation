from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adadelta

import matplotlib.pyplot as plt

import sys
import glob

import numpy as np
import pickle as pkl

class Generator():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_joints = 15

        optimizer = Adam()

        # Build and compile the network
        self.network = self.build_network()
        self.network.compile(loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.network.summary()

    def build_network(self):

        inputs = Input((self.img_rows, self.img_cols, self.channels))

        encoder = self.build_encoder()(inputs)

        return Model(inputs=inputs, outputs=encoder)

    def build_encoder(self):
        model = Sequential(name='Encoder')

        # Encoder model
        model.add(Conv2D(64, kernel_size=4, padding='same', input_shape=(self.img_rows, self.img_cols, self.channels)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=4, padding='same'))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(3 * self.num_joints))
        model.add(Activation('linear'))

        return model

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Read data
        files = sorted(glob.glob("cropped_images.pkl"))

        for file in files:
            if 'X1' in locals():
                X1 = np.concatenate((X1, pkl.load(open(file, "rb"))))
            else:
                X1 = pkl.load(open(file, "rb"))
        X1 = (X1.astype(np.float32) * 2 - np.amax(X1)) / np.amax(X1)
        self.train_data = X1

        files = sorted(glob.glob("train_real_labels.pkl"))

        for file in files:
            if 'X2' in locals():
                X2 = np.concatenate((X2, pkl.load(open(file, "rb"))))
            else:
                X2 = pkl.load(open(file, "rb"))
        self.train_labels = X2

        # Data normalization
        self.train_labels = np.reshape(self.train_labels, (self.train_labels.shape[0], self.train_labels.shape[1]))
        self.train_labels = np.delete(self.train_labels, [27, 28, 29, 30, 31, 32, 45, 46, 47, 48, 49, 50, 51, 52, 53], axis=1)

        print(np.amin(self.train_data))
        print(np.amax(self.train_data))

        history_callback = self.network.fit(x=self.train_data, y=self.train_labels, epochs=epochs, validation_split=0.1)
        loss = np.array(history_callback.history["loss"])
        v_loss = np.array(history_callback.history["val_loss"])
        np.savetxt("loss.txt", loss, delimiter=",")
        np.savetxt("validation_loss.txt", v_loss, delimiter=",")

        self.save_network();

    def save_network(self):
        print("Saving network")
        self.network.save("network.hd5")


if __name__ == '__main__':
    gen = Generator()
    gen.train(epochs=30, batch_size=32, sample_interval=200)

