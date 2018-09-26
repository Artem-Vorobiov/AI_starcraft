import keras  # Keras 2.1.2 and TF-GPU 1.8.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random

#	Смотрю принт и Тип каждоый строки
#	Смотрю что делает каждая функция и ее аргументы
# 	После вновь смотрю принцип работы ЭТОЙ Сети

model = Sequential()
# print('\n MODEL \n', model)			#	<keras.models.Sequential object at 0x103dba3c8>
# print('\n MODEL \n', type(model))		#	<class 'keras.models.Sequential'>


#	-- Class tf.keras.layers.Conv2D  AND tf.keras.layers.Convolution2D
#	-- conv2d(). Constructs a two-dimensional convolutional layer. +
# Takes number of filters, filter kernel size, padding, and activation function as arguments.
#	-- input_shape=(176, 200, 3) - ВХОДНАЯ ФОРМА(РАЗМЕР) -  мы используем именно эти данные +
# так как это размер наших картинок из - game_info.map_size()
#	-- padding: one of "valid" or "same" (case-insensitive).
#	-- kernel_size = (3,3)

one = model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3),
                 activation='relu'))
# print('\n Conv2D \n', one)				#	None
# print('\n Conv2D \n', type(one))			#	<class 'NoneType'>

model.add(Conv2D(32, (3, 3), activation='relu'))
#	-- max_pooling2d(). Constructs a two-dimensional pooling layer using the max-pooling algorithm. 
# Takes pooling filter size and stride as arguments.
model.add(MaxPooling2D(pool_size=(2, 2)))
#	-- keras.layers.Dropout(rate, noise_shape=None, seed=None)
#	-- Dropout consists in randomly setting a fraction rate of input units to
#  0 at each update during training time, which helps prevent overfitting.
model.add(Dropout(0.2))
  

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#	-- keras.layers.Flatten(data_format=None)
#	-- Flattens the input. Does not affect the batch size.
model.add(Flatten())

#	-- dense(). Constructs a dense layer. Takes number of neurons and activation function as arguments.
#	-- keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
#	-- bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
#	-- kernel_constraint=None, bias_constraint=None)
#	-- Just your regular densely-connected NN layer.

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)
# print('\n opt \n', opt)					#	<keras.optimizers.Adam object at 0x11ec3e5c0>
# print('\n opt \n', type(opt))				#	<class 'keras.optimizers.Adam'>

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/stage1")
print('\n tensorboard \n', tensorboard)			#	<keras.callbacks.TensorBoard object at 0x11ec3e320>
print('\n tensorboard \n', type(tensorboard))	#	<class 'keras.callbacks.TensorBoard'>

print('\n IN PROCESS ')
