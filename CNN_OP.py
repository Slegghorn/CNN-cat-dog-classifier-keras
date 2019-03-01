import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import time

x = np.load('x_data.npy')
y = np.load('y_data.npy')
x = x/255.0

NAME = 'CNN--cat-dog'
tensorboard = TensorBoard(log_dir = './logs/{}'.format(NAME))

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape = x.shape[1:], activation = tf.nn.relu))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = tf.nn.relu))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = tf.nn.relu))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(1, activation = tf.nn.sigmoid))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x, y, batch_size = 32, validation_split = 0.1, epochs = 10, callbacks = [tensorboard])

model.save('CNN_cat-dog.h5')
