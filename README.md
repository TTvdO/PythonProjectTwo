How to run the code:
1. Install anaconda on your PC (https://www.anaconda.com/products/individual)
2. Make an anaconda environment
3. Install tensorflow into the anaconda environment
4. Install all necessary packages (e.g. numpy, sklearn, ...) into the anaconda environment 
5. Download the folder of images of dogs and cats our convolutional neural network will use from here: https://www.microsoft.com/en-us/download/details.aspx?id=54765
6. Enter the correct path to the folder with both the dog and cat images in them in the load_images_and_save class
7. Onto the code, first run the load_images_and_save class to load in all the images into data (features and labels) that our neural network can use as input
8. Then run the create_and_train_ConvNeuralNetwork class to create a Convolutional Neural Network model trained with the image data
9. Then run the cats_vs_dogs_predicting class to test the accuracy on the test data (out of sample data, unknown to the algorithm) that was seperated from the training data to test the accuracy of the Convolutional Neural Network and how well this network can generalize
10. If you want to try to increase the accuracy of the model, you can use a process called hyperparameter tuning which is done in the hyperparameter_tuning class. This class will try out all the different combinations of the available parameters specified in the class itself. Just know, this takes a long time. And of course, to optimize it further, much more than just hyperparameter tuning would have to be done and the amount of hyperparameters to consider would be far greater. Plus, Bayesian Optimization is probably best for hyperparameter tuning in most cases, which the package I've chosen doesn't do. Nonetheless, it is a decent start.