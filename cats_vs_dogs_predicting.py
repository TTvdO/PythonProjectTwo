import cv2
import tensorflow as tf
import numpy as np

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

model = tf.keras.models.load_model("ConvolutionalNeuralNetworkV1.model")

# correctPredictions = 0
# for index, pixelArray in enumerate(X_test):
#     # how the model.predict function wants its data depends on the model itself. check how the first layer of the neural network wants its data
#     # could also change the input shape potentially, but this is probably not ideal. you want to give the algorithm 48x48 pixels,
#     # in this case these pixels are represented by every array in X_test[index]
#     prediction = int(model.predict(X_test[index]))
#     if(prediction == y_test[index]):
#         correctPredictions += 1

correctPredictions = 0
for index, pixelArrays in enumerate(X_test):
    # how the model.predict function wants its data depends on the model itself. check how the first layer of the neural network wants its data
    # could also change the input shape potentially, but this is probably not ideal. you want to give the algorithm 48x48 pixels,
    # in this case these pixels are represented by every array in X_test[index]
    arraysToPredict = []
    for pixelArray in pixelArrays[index]:
        arraysToPredict.append(pixelArray)

    arraysToPredict = np.array(arraysToPredict)
    arraysToPredict = np.concatenate(arraysToPredict).ravel()
    arraysToPredict = np.array(arraysToPredict).reshape(-1, 48, 48, 1)
    prediction = int(model.predict(arraysToPredict))
    if(prediction == y_test[index]):
        correctPredictions += 1

# prediction = int(model.predict(X_test))

outOfSampleAccuracy = correctPredictions / len(X_test)
print(outOfSampleAccuracy)