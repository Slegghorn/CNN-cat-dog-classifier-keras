import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]
x = np.load('x_data.npy')
y = np.load('y_data.npy')
x = x/255.0

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir = './logs/{}'.format(NAME))

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape = x.shape[1:], activation = tf.nn.relu))
            model.add(MaxPooling2D(pool_size = (2, 2)))
            for i in range(conv_layer - 1):
                model.add(Conv2D(layer_size, (3, 3), activation = tf.nn.relu))
                model.add(MaxPooling2D(pool_size = (2, 2)))

            model.add(Flatten())
            for i in range(dense_layer):
                model.add(Dense(layer_size, activation = tf.nn.relu))
            model.add(Dense(1, activation = tf.nn.sigmoid))

            model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

            model.fit(x, y, batch_size = 32, validation_split = 0.3, epochs = 10, callbacks = [tensorboard])
