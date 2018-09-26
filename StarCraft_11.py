import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3),
                 activation='relu'))
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

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/STAGE1")

train_data_dir = "train_data"
count = 0 

def check_data():
    choices = {"no_attacks": no_attacks,
               "attack_closest_to_nexus": attack_closest_to_nexus,
               "attack_enemy_structures": attack_enemy_structures,
               "attack_enemy_start": attack_enemy_start}

    # print("\n CHOICES = {} ", choices)                  #   'no_attacks' : [ChoiseArray and PicturesArray]
    # print("\n CHOICES--TYPE = {} ", type(choices))      #   <class 'dict'>

    total_data = 0

    lengths = []
    for choice in choices:
        # print("\n CHOICE = {} ", choice)                                        # no_attacks
        # print("\n CHOICE = {} ", type(choice))                                  # <class 'str'>
        print("Length of {} is: {}".format(choice, len(choices[choice])))         #   Length of no_attacks is: 45
        # print("\n choices[choice] = {} ", choices[choice])                      #   [ChoiseArray and PicturesArray]
        # print("\n choices[choice] = {} ", type(choices[choice]))                #   <class 'list'>
        total_data += len(choices[choice])
        print("\n total_data = {} ", total_data)                                  #     154
        print("\n total_data = {} ", type(total_data))                            #     <class 'int'>

        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths


hm_epochs = 10

for i in range(hm_epochs):
    print("\n Currnet Epoch = {} ", i)
    current = 0
    increment = 200
    not_maximum = True
    all_files = os.listdir(train_data_dir)
    # print("\n all_files = {} ", all_files)              #    ['1530807965.npy', '1530808170.npy', '1530808780.npy', '1530809759.npy']
    # print("\n all_files--TYPE = {} ", type(all_files))  #    <class 'list'>

    maximum = len(all_files)                            
    random.shuffle(all_files)                           
    # print("\n maximum = {} ", maximum)                  #   Max = 4
    # print("\n maximum--TYPE = {} ", type(maximum))      #   <class 'int'>

# print("\n maximum = {} ", maximum)
# print("\n maximum--TYPE = {} ", type(maximum))

    while not_maximum:
        print("WORKING ON {}:{}".format(current, current+increment))
        no_attacks = []
        attack_closest_to_nexus = []
        attack_enemy_structures = []
        attack_enemy_start = []

        # print("\n all_files[current:current+increment] = {} ", all_files[current:current+increment])
        # print("\n all_files[current:current+increment]--TYPE = {} ", type(all_files[current:current+increment]))
#   all_files[current:current+increment] --  <class 'list'> --> ['1530807965.npy', '1530808170.npy', '1530808780.npy', '1530809759.npy']

        for file in all_files[current:current+increment]:
            # print("\n file = {} ", file)                    #   1530808170.npy
            # print("\n file--TYPE = {} ", type(file))        #   <class 'str'>
            full_path = os.path.join(train_data_dir, file)
            # print("\n full_path = {} ", full_path)              #   train_data/1530809759.npy
            # print("\n full_path--TYPE = {} ", type(full_path))  #   <class 'str'>
            data = np.load(full_path)
            # print("\n data = {} ", data)                        #   Просто массив
            # print("\n data--TYPE = {} ", type(data))            #   <class 'numpy.ndarray'>
            data = list(data)
            # print("\n data = {} ", data)                        #    Гиганский массив в который входит два массива -  ChoiseArray and PicturesArray
            # print("\n data--TYPE = {} ", type(data))            #   <class 'list'>
            for d in data:
                # print("\n d = {} ", d)                      #   d - это или ChoiseArray или PicturesArray
                # print("\n d--TYPE = {} ", type(d))          #   <class 'numpy.ndarray'>

                #   Определяет максимальный аршумент в массиве. В нашем случае где номер 1 и возвращает его порядковый номер
                choice = np.argmax(d[0])
                # print("\n choice = {} ", choice)                # 1 
                # print("\n choice--TYPE = {} ", type(choice))    # <class 'numpy.int64'>                    
                if choice == 0:
                    no_attacks.append([d[0], d[1]])
                elif choice == 1:
                    attack_closest_to_nexus.append([d[0], d[1]])
                elif choice == 2:
                    attack_enemy_structures.append([d[0], d[1]])
                elif choice == 3:
                    attack_enemy_start.append([d[0], d[1]])
                count += 1
                print('\n   START.', count, '\n')
                # print("\n no_attacks = {} ", len(no_attacks))
                # print("\n attack_closest_to_nexus = {} ", len(attack_closest_to_nexus))
                # print("\n attack_enemy_structures = {} ", len(attack_enemy_structures))
                # print("\n attack_enemy_start = {} ", len(attack_enemy_start))
                # print("\n attack_enemy_start--TYPE = {} ", type(attack_enemy_start))    #   <class 'list'>

                #   Файл npy  состоит из N-ного колличества массивов, в одном таком массиве два подмассива --
                #   Первый подмассив это ChoiseArray, второй подмассив это PicturesArray
                #   Создаем четрые ЛИста-Массива no_attacks, attack_closest_to_nexus, attack_enemy_structures, attack_enemy_start
                #   Мы добираемся до ChoiseArray, определяем где 1 и заносим эго(и PicturesArray) в один из 4-х Листов-Массивов

        lengths = check_data()
        # print("\n lengths  ", lengths)                  #   [45, 34, 42, 33]
        # print("\n lengths TYPE", type(lengths))         #   <class 'list'>
        lowest_data = min(lengths)
        # print("\n lowest_data  ", lowest_data)            #   33
        # print("\n lowest_data TYPE", type(lowest_data))   #   <class 'int'>

        random.shuffle(no_attacks)
        random.shuffle(attack_closest_to_nexus)
        random.shuffle(attack_enemy_structures)
        random.shuffle(attack_enemy_start)

        no_attacks = no_attacks[:lowest_data]
        # print("\n no_attacks[:lowest_data]  ", no_attacks)                   #   [ChoiseArray and PicturesArray]
        # print("\n no_attacks[:lowest_data] TYPE", type(no_attacks))          #   <class 'list'>
        attack_closest_to_nexus = attack_closest_to_nexus[:lowest_data]
        attack_enemy_structures = attack_enemy_structures[:lowest_data]
        attack_enemy_start = attack_enemy_start[:lowest_data]

        check_data()

        train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start
        print("\n train_data ", train_data)                    #   [ChoiseArray and PicturesArray]
        print("\n train_data TYPE", type(train_data))          #   <class 'list'>

        random.shuffle(train_data)
        print(len(train_data))

        test_size = 100
        batch_size = 128

        # x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3)
        # y_train = np.array([i[0] for i in train_data[:-test_size]])

        # x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
        # y_test = np.array([i[0] for i in train_data[-test_size:]])

        # model.fit(x_train, y_train,
        #           batch_size=batch_size,
        #           validation_data=(x_test, y_test),
        #           shuffle=True,
        #           verbose=1, callbacks=[tensorboard])

        # model.save("BasicCNN-{}-epochs-{}-LR-STAGE1".format(hm_epochs, learning_rate))
        # current += increment
        # if current > maximum:
        #     not_maximum = False