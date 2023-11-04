import pandas as pd
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout #dense layer
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

predictors = pd.read_csv('C:\\Users\\mateu\\MACHINE_LEARNING_ENPC\\NEURAL_NETWORKS\\BREAST_CANCER\\input_breast.csv')
classes = pd.read_csv('C:\\Users\\mateu\\MACHINE_LEARNING_ENPC\\NEURAL_NETWORKS\\BREAST_CANCER\\output_breast.csv')

def create_net(optimizer, loss, activation, neurons = 16, kernel_initializer = 'random_uniform'):
    net = Sequential()
    net.add(Dense(units = neurons,
                            activation = activation,
                            kernel_initializer=kernel_initializer,
                            input_dim = 30))
    net.add(Dropout(0.2))
    net.add(Dense(units = neurons,
                            activation=activation,
                            kernel_initializer=kernel_initializer))
    net.add(Dropout(0.2))
    net.add(Dense(units = 1,
                activation='sigmoid'))
    net.compile(optimizer,loss,metrics=['binary_accuracy'])
    return net

net = KerasClassifier(build_fn=create_net)

parameters = {'batch_size': [10,30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'activation': ['relu', 'tanh']
              #'kernel_initializer': ['random_uniform', 'normal'],
              #'neurons': [16,8]
              }

grid_search = GridSearchCV(estimator = net, param_grid=parameters,
                        #    scoring = 'accuracy', cv = 2,
                             error_score='raise')
grid_search.fit(predictors, classes)

print(f"Best Parameters: {grid_search.best_params_}\nBest Score: {grid_search.best_score_}")