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


#	Class tf.keras.layers.Conv2D  AND tf.keras.layers.Convolution2D
#	input_shape=(176, 200, 3) - мы используем именно эти данные так как это размер наших картинок из - game_info.map_size()
one = model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3),
                 activation='relu'))
# print('\n Conv2D \n', one)				#	None
# print('\n Conv2D \n', type(one))			#	<class 'NoneType'>

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
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


model.add(Flatten())
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
