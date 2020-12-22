import tensorflow as tf
import numpy as np

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

model = tf.keras.models.load_model("ConvolutionalNeuralNetworkV1.model")

listOfPredictions = model.predict(X_test)

correctPredictions = 0
for index, prediction in enumerate(listOfPredictions):
    predictionClass = int(round(prediction[0]))
    if(predictionClass == y_test[index]):
        correctPredictions += 1

outOfSampleAccuracy = correctPredictions / len(X_test)
print(outOfSampleAccuracy)