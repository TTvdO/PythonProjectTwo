import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import cv2
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import pickle
import random

IMAGESDIRECTORY = "C:/Users/Tim/Desktop/PetImages"
IMG_SIZE = 48
CATEGORIES = ["Dog", "Cat"]

petImages = []

def load_images():
    for category in CATEGORIES:
        # path = os.path.join(IMAGESDIRECTORY, category)
        path = f'{IMAGESDIRECTORY}/{category}'
        classLabel = CATEGORIES.index(category)
        currentImageFolder = os.listdir(path)
        counter = 0
        # loop through everything except for the last element (last element is a .db file)
        for petImage in currentImageFolder[:-1]:
            try:
                petImage = image.imread(f'{IMAGESDIRECTORY}/{category}/{counter}.jpg')
                petImage = cv2.cvtColor(petImage, cv2.COLOR_RGB2GRAY)
                resizedPetImage = cv2.resize(petImage, (IMG_SIZE, IMG_SIZE))
                petImages.append([resizedPetImage, classLabel])
            except Exception as e:
                pass
            counter += 1

load_images()

random.shuffle(petImages)

X = []
y = []

for features, label in petImages:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X/255.0

np.save('features.npy', X)
np.save('labels.npy', y)