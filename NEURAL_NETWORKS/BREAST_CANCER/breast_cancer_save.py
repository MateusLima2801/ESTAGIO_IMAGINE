import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout #dense layer
import numpy as np

predictors = pd.read_csv('C:\\Users\\mateu\\MACHINE_LEARNING_ENPC\\NEURAL_NETWORKS\\BREAST_CANCER\\input_breast.csv')
classes = pd.read_csv('C:\\Users\\mateu\\MACHINE_LEARNING_ENPC\\NEURAL_NETWORKS\\BREAST_CANCER\\output_breast.csv')

net = Sequential()
net.add(Dense(units = 8, activation ='relu',
                            kernel_initializer='normal',
                            input_dim = 30))
net.add(Dropout(0.2))
net.add(Dense(units = 8, activation='relu',
                            kernel_initializer='normal'))
net.add(Dropout(0.2))
net.add(Dense(units = 1,activation='sigmoid'))
net.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics=['binary_accuracy'])

net.fit(predictors, classes, batch_size=10, epochs=100)

net_json = net.to_json()

with open("net_breast.json", "w") as json_file:
    json_file.write(net_json)

net.save_weights('net_breast.h5')