import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import talos

X = np.load('features.npy')
y = np.load('labels.npy')

params = {
    'first_neuron_amount': [16, 32, 64, 128, 256],
    'second_neuron_amount': [16, 32, 64, 128, 256],
    'third_neuron_amount': [16, 32, 64, 128, 256],
    'first_activation': ['relu', 'sigmoid', 'elu', 'softmax', 'tanh'],
    'second_activation': ['relu', 'sigmoid', 'elu', 'softmax', 'tanh'],
    'third_activation': ['relu', 'sigmoid', 'elu', 'softmax', 'tanh'],
    'first_poolingsize': [(1,1), (2,2)],
    'second_poolingsize': [(1,1), (2,2)],
    'batch_size': [8, 16, 32, 64]
}

def catsvsdogs(x_train2, y_train2, x_test2, y_test2, params):
    model = Sequential()

    model.add(Conv2D(params['first_neuron_amount'], (3, 3), input_shape = (X.shape[1:])))
    model.add(Activation(params['first_activation']))
    model.add(MaxPooling2D(pool_size=params['first_poolingsize']))

    model.add(Conv2D(params['second_neuron_amount'], (3, 3)))
    model.add(Activation(params['second_activation']))
    model.add(MaxPooling2D(pool_size=params['second_poolingsize']))

    model.add(Flatten())

    model.add(Dense(params['third_neuron_amount']))
    model.add(Activation(params['third_activation']))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    out = model.fit(x_train2, y_train2, batch_size=params['batch_size'], epochs=3)

    return out, model

t = talos.Scan(x=X, y=y, params=params, model=catsvsdogs, experiment_name='test_talos3')