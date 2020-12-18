import cv2
import tensorflow as tf
import numpy as np

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

model = tf.keras.models.load_model("ConvolutionalNeuralNetworkV1.model")

correctPredictions = 0
for index, test_features in enumerate(X_test):
    prediction = int(model.predict(test_features))
    if(prediction == y_test[index]):
        correctPredictions += 1

outOfSampleAccuracy = correctPredictions / len(X_test)
print(outOfSampleAccuracy)