import cv2
import tensorflow as tf
import numpy as np

# CATEGORIES = ["Cat, Dog"]

# def prepare(filepath):
#     IMG_SIZE=100
#     img_array=cv2.imread(filepath)
#     img_array = img_array/255.0
#     new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
#     return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_test = X_test.reshape(-1, 48, 48, 1)

model = tf.keras.models.load_model("ConvolutionalNeuralNetworkV1.model")

correctPredictions = 0
for index, test_features in enumerate(X_test):
    prediction = int(model.predict(test_features))
    if(prediction == y_test[index]):
        correctPredictions += 1

accuracy = correctPredictions / len(X_test)
print(accuracy)

# prediction = model.predict([prepare('dog.jpg')])
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict([prepare('cat.jpg')])
# print(CATEGORIES[int(prediction[0][0])])