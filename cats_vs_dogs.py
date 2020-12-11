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


# steps:

# DATA
# -reference and store petimages folder in a variable
# -loop through pet images by finding a way to reference the Cat folder and the Dog folder respectively. anything in the Cat folder gets the label 0, Dogs get label 1
    # p.s. you need to add to the already existing path to the petimages folder here. you'd need to add /Cat and /Dog respectively
# -make sure all the images are the same size by hardcoding the size to e.g. 50 by 50, just execute this step at the beginning of the loop
# -shuffle the data
# -assign variable X to all the attributes besides the label(the features), assign variable to all the labels of the data
# -figure out how to format this data in order to pass it onto the algorithm that will classify the data

# temp:
# ok so I have the images in lists. now I want to:
# -transform all these images to the same size, probably before adding them to the list 
# -figure out what it is exactly that you want to use as your features. I guess all of the pixels of the 50 by 50 pixels. 
# -add the label as a last element OR create a dictionary containing a list of the features as its first element 
# -find a way to display any of the images on your screen (catImages[0] for example)

# get the cat and dog images out of the pickle state here
with open('catAndDogImagesWithLabels', 'rb') as f:
    catImagesWithLabels, dogImagesWithLabels = pickle.load(f)

print(len(catImagesWithLabels))
print(len(dogImagesWithLabels))

# ALGORITHM
# -fit the algorithm with the data (train), using the X_train and y_train data
# -predict the list of all the features, compared to the actual label values. a.k.a. use the predict method in which you pass the X_test and y_test
# 