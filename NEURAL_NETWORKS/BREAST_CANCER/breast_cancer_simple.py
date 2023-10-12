import pandas as pd
from sklearn.model_selection import train_test_split
import keras 
from keras.models import Sequential
from keras.layers import Dense #dense layer
from sklearn.metrics import confusion_matrix, accuracy_score

predictors = pd.read_csv('C:\\Users\\mateu\\MACHINE_LEARNING_ENPC\\NEURAL_NETWORKS\\BREAST_CANCER\\input_breast.csv')
classes = pd.read_csv('C:\\Users\\mateu\\MACHINE_LEARNING_ENPC\\NEURAL_NETWORKS\\BREAST_CANCER\\output_breast.csv')

predictors_training, predictors_test, classes_training, classes_test = train_test_split(predictors, classes, test_size = 0.25)

#let's do a perceptron, starting with hidden layer size as the mean of input and output size:
# input 30
# output 1
# mean ~ 16
net = Sequential()
net.add(Dense(units = 16,
                        activation='relu',
                        kernel_initializer='random_uniform',
                        input_dim = 30))
net.add(Dense(units = 16,
                        activation='relu',
                        kernel_initializer='random_uniform'))
net.add(Dense(units = 1,
              activation='sigmoid'))

optimizer = keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.001, clipvalue=0.5)

net.compile(optimizer,
            loss = 'binary_crossentropy',
            metrics=['binary_accuracy'])

net.fit(predictors_training, classes_training, batch_size=10, epochs = 100)

weights0 = net.layers[0].get_weights()
print(weights0)

weights1 = net.layers[1].get_weights()
print(weights1)

weights2 = net.layers[2].get_weights()
print(weights2)

# evaluate using keras
predictions = net.predict(predictors_test)
predictions = (predictions > 0.5) #binary
accuracy = accuracy_score(classes_test, predictions)
conf_matrix = confusion_matrix(classes_test, predictions)

print(conf_matrix)

# evaluate using sklearn: loss, binary_accuracy
result = net.evaluate(predictors_test, classes_test)