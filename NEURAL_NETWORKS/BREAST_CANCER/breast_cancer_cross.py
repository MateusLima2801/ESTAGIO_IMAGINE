import pandas as pd
from sklearn.model_selection import train_test_split
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout #dense layer
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

predictors = pd.read_csv('C:\\Users\\mateu\\MACHINE_LEARNING_ENPC\\NEURAL_NETWORKS\\BREAST_CANCER\\input_breast.csv')
classes = pd.read_csv('C:\\Users\\mateu\\MACHINE_LEARNING_ENPC\\NEURAL_NETWORKS\\BREAST_CANCER\\output_breast.csv')

def create_net():
    net = Sequential()
    net.add(Dense(units = 16,
                            activation='relu',
                            kernel_initializer='random_uniform',
                            input_dim = 30))
    net.add(Dropout(0.2))
    net.add(Dense(units = 16,
                            activation='relu',
                            kernel_initializer='random_uniform'))
    net.add(Dropout(0.2))
    net.add(Dense(units = 1,
                activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.001, clipvalue=0.5)

    net.compile(optimizer,
                loss = 'binary_crossentropy',
                metrics=['binary_accuracy'])
    return net

net = KerasClassifier(build_fn=create_net,
                      epochs = 100,
                      batch_size = 10)

results = cross_val_score(estimator=net,
                          X = predictors, y = classes,
                          cv=2, scoring='accuracy')

mean = results.mean()
deviation = results.std()

print(f"Mean: {mean}, Deviation: {deviation}")