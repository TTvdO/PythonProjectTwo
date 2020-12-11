import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import cv2
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import pickle

catImagesFolder = os.listdir("C:/Users/Tim/Desktop/PetImages/Cat")
dogImagesFolder = os.listdir("C:/Users/Tim/Desktop/PetImages/Dog")

catImages = []
dogImages = []

counter = 0

for catImage in catImagesFolder[:-1]:
    try:
        catImage = image.imread(f'C:/Users/Tim/Desktop/PetImages/Cat/{counter}.jpg')
        catImage = cv2.cvtColor(catImage, cv2.COLOR_RGB2GRAY)
        catImageResized = cv2.resize(catImage, (48,48))
        catImages.append(catImageResized)
    except Exception as e:
        pass
    counter += 1
counter = 0
for dogImage in dogImagesFolder[:-1]:
    try:
        dogImage = image.imread(f'C:/Users/Tim/Desktop/PetImages/Dog/{counter}.jpg')
        dogImage = cv2.cvtColor(dogImage, cv2.COLOR_RGB2GRAY)
        dogImageResized = cv2.resize(dogImage, (48,48))
        dogImages.append(dogImageResized)
    except Exception as e:
        pass
    counter += 1

catImages = np.array(catImages)
dogImages = np.array(dogImages)

with open('catAndDogImages', 'wb') as f:
    pickle.dump([catImages, dogImages], f)