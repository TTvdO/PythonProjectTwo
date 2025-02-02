import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

X = np.load('features.npy')
y = np.load('labels.npy')

# split the data up into training and testing data (in sample & out of sample data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# so we can access the X_test and y_test data in our prediction class later. we need the same randomly chosen X_test and y_test values, because the algorithm
# shouldn't have already seen the images that we are testing it on, a.k.a. we can't randomly shuffle the data again in the class where we predict, because then
# images in our new testing data will have actually been part of our training data in this class
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# make sequential model, which we can add layers to and thus create the convolutional neural network we need to classify images
model = Sequential()

# layer 1, input layer
# 1. first argument Conv2D: filters
# 2. second argument Conv2D: kernel size. must be a tuple of odd integers with the same values. if input image > 128x128, consider (5,5) and (7,7)
#       if input image < 128x128, consider (1,1) and (3,3)
# 3. input_shape, which are the WxH of the image together with the amount of numbers representing each pixel (e.g. 3 for RGB images, 1 for grayscale images)
model.add(Conv2D(32, (3, 3), input_shape = (X.shape[1:])))
# activation function "rectified linear" does this: negative number as input -> return 0 as output. positive number -> return the actual number as output
# even though all the values of pixels aren't going to be negative, even when divided by 255, the nodes in each layer of the neural network will have not only
# the inputs, but also their own randomly initialized weights. the inputs will be multiplied by these weights and could make your output negative. 
# to avoid this we use this rectified linear activation function that returns 0 as output for any negative number
model.add(Activation("relu"))
# max pooling helps reduce the size of the output volume. can also be achieved by increasing an optional parameter of Conv2D 'strides' to (2,2)
# maxpooling converges 2 values into 1. images are a unique situation where this is acceptable, since pixels one to the right or left will often be roughly the same.
model.add(MaxPooling2D(pool_size=(2,2)))

# layer 2
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten converts the 3D feature maps to 1D feature vectors, which is necessary for the next dense layer that only works with 1D data
model.add(Flatten())

# layer 3
# dense layer just states: this is how many inputs we are expecting for this layer. it will use (input * weight) for the output value of every single node
model.add(Dense(64))
model.add(Activation("relu"))

# layer 4, output layer
model.add(Dense(1))
# return value between 0 and 1 as an output. e.g. value of .74 for a classification problem means the algorithm thinks the thing to predict was a 1 as class most likely.
# you'd just convert the prediction to an integer at the time of predicting new images with this model to get a 0 or 1
model.add(Activation("sigmoid"))

# compile, which readies the model for training
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=10)

model.save('ConvolutionalNeuralNetworkV1.model')