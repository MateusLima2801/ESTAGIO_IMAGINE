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

new  = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])
prediction = net.predict(new)
prediction = (prediction > 0.5)
print(prediction)
