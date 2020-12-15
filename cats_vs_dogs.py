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

# steps:

# DATA
# -[X]reference and store petimages folder in a variable
# -[X]loop through pet images by finding a way to reference the Cat folder and the Dog folder respectively. anything in the Cat folder gets the label 0, Dogs get label 1
    # p.s. you need to add to the already existing path to the petimages folder here. you'd need to add /Cat and /Dog respectively
# -[X]make sure all the images are the same size by hardcoding the size to e.g. 50 by 50, just execute this step at the beginning of the loop
# -[X]shuffle the data
# -[X]assign variable X to all the attributes besides the label(the features), assign variable to all the labels of the data
# -[X]split your X and y into X_test, y_test, X_train, y_train

# get the X data (features/input data) and y data (labels/targets) out of the pickle state here
with open('Xy', 'rb') as f:
    X, y = pickle.load(f)

# split the data up into training and testing data (in sample & out of sample data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# now you have the data that you need. now it's time to figure out:
# -[]how to pass this data properly to an algorithm
        # Xy is a numpy array filled with arrays right now. depending on the algorithm you're going to use, you might need to transform this data into a single array, 
        # but you'd need to know which algorithm you're using first and find out how it wants to receive the data through scikit-learn docs and stackoverflow threads
# -[X]which algorithm to select for training:
    # Convolutional Neural Networks are the correct choice for image classification problems
    # TODO: now to find out in which form this algorithm wants its data. I think it wants it as one big array, so you'd need to just put all the arrays together
    # for every single 2d array, and then move on to the stuff listed under algorithm 

# ALGORITHM
# -create the model outline
# -fit the algorithm with the data (train), using the X_train and y_train data
# -predict the list of all the features, compared to the actual label values. a.k.a. use the predict method in which you pass the X_test and y_test
# 