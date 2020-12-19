import cv2
import tensorflow as tf
import numpy as np

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

model = tf.keras.models.load_model("ConvolutionalNeuralNetworkV1.model")

scores = model.evaluate(X_test, y_test)
print(scores)