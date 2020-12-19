# python project two = classifying cats vs dogs with machine learning
# import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import cv2
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import pickle
from sklearn.model_selection import train_test_split #, cross_validate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.activations import relu, elu
import tensorflow as tf
import talos

X = np.load('features.npy')
y = np.load('labels.npy')

params = {
    'neuron_amount': [16, 32, 64, 128, 256],
    'activation': ['relu', 'sigmoid', 'elu', 'softmax', 'tanh'],
    'poolingsize': [(1,1), (2,2)],
    'loss': ['binary_crossentropy', 'sparse_categorical_crossentropy'],
    # 'optimizer': ['adam', 'sgd'],
    'batch_size': [8, 16, 32, 64]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(len(X_train))
print(len(y_train))

# np.save('X_test.npy', X_test)
# np.save('y_test.npy', y_test)

# model.save('ConvolutionalNeuralNetworkV2.model')

def catsvsdogs(x_train2, y_train2, x_test2, y_test2, params):
    model = Sequential()

    model.add(Conv2D(params['neuron_amount'], (3, 3), input_shape = (X.shape[1:])))
    model.add(Activation(params['activation']))
    model.add(MaxPooling2D(pool_size=params['poolingsize']))

    model.add(Conv2D(params['neuron_amount'], (3, 3)))
    model.add(Activation(params['activation']))
    model.add(MaxPooling2D(pool_size=params['poolingsize']))

    model.add(Flatten())

    model.add(Dense(params['neuron_amount']))
    model.add(Activation(params['activation']))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss=params['loss'], optimizer='adam', metrics=['accuracy'])

    out = model.fit(X_train, y_train, batch_size=32, epochs=10)

    return out, model

t = talos.Scan(x=X, y=y, params=params, model=catsvsdogs, experiment_name='test_talos')
# t = talos.Scan(x=X_train, y=y_train, params=params, model=catsvsdogs, experiment_name='test_talos', x_val=X_test, y_val=y_test, val_split=0.0, shuffle=False)