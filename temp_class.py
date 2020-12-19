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
import tensorflow as tf

X = np.load('features.npy')
y = np.load('labels.npy')

# split the data up into training and testing data (in sample & out of sample data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# so we can access the X_test and y_test data in our prediction class later. we need the same randomly chosen X_test and y_test values, because the algorithm
# shouldn't have already seen the images that we are testing it on, a.k.a. we can't randomly shuffle the data again in the class where we predict, because then
# images in our new testing data will have actually been part of our training data in this class
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)