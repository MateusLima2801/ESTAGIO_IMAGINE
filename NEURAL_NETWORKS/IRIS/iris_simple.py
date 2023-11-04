import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import numpy as np
base = pd.read_csv('NEURAL_NETWORKS\\IRIS\\iris.csv')
predictors = base.iloc[:, 1:4].values
classes = base.iloc[:,4].values

le = LabelEncoder()
classes = le.fit_transform(classes)
classes_dummy = []
repo_classes = [np.array([1,0,0]),
                np.array([0,1,0]),
                np.array([0,0,1])]
for c in classes:
    classes_dummy.append(repo_classes[c])
classes_dummy = np.array(classes_dummy)

predictors_training, predictors_test, classes_training, classes_test = train_test_split(predictors, classes_dummy, test_size = 0.33)

# output (1,0,0) (0,1,0) (0,0,1)
net = Sequential()
#units = input + output/ 2 ~ 4 (first guess)
net.add(Dense(units=4, activation = 'relu', input_dim = 4))
net.add(Dense(units=4, activation = 'relu', input_dim = 4))
# classification with more than 2 classes is better made with softmax
# probability output
net.add(Dense(units=3, activation = 'softmax'))
net.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'] )

net.fit(predictors_training, classes_training, batch_size=10, epochs=1000)