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
petImages = []

counter = 0

# hier al doen:
# -gemixte lijst van beide cat en dog images, die je daarna nog shuffled[x]
# -aparte lijsten voor X en y scheiden van elkaar
for catImage in catImagesFolder[:-1]:
    try:
        catImage = image.imread(f'C:/Users/Tim/Desktop/PetImages/Cat/{counter}.jpg')
        catImage = cv2.cvtColor(catImage, cv2.COLOR_RGB2GRAY)
        catImageResized = cv2.resize(catImage, (48,48))
        petImages.append([0, catImageResized])
    except Exception as e:
        pass
    counter += 1
counter = 0
for dogImage in dogImagesFolder[:-1]:
    try:
        dogImage = image.imread(f'C:/Users/Tim/Desktop/PetImages/Dog/{counter}.jpg')
        dogImage = cv2.cvtColor(dogImage, cv2.COLOR_RGB2GRAY)
        dogImageResized = cv2.resize(dogImage, (48,48))
        petImages.append([1, dogImageResized])
    except Exception as e:
        pass
    counter += 1
petImages = np.array(petImages)
np.random.shuffle(petImages)
print(petImages[1])
X = petImages[:,0]
print(X[1])
y = petImages[:,1]
print(y[1])
# catImages = np.array(catImages)
# dogImages = np.array(dogImages)

with open('Xy2', 'wb') as f:
    pickle.dump([X, y], f)