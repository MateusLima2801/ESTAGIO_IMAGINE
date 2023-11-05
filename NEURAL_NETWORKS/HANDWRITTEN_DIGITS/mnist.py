import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization.batch_normalization import BatchNormalization

#mnist has 28x28 images then 784 pixels

(x_train, y_train), (x_test, y_test) = mnist.load_data()
predictors_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
predictors_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

classes_train = np_utils.to_categorical(y_train, 10)
classes_test = np_utils.to_categorical(y_test, 10)

classificator = Sequential()
#32 kernels, recommended 64
classificator.add(Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'))
classificator.add(BatchNormalization())
classificator.add(MaxPooling2D(pool_size=(2,2)))

#32 kernels, recommended 64
classificator.add(Conv2D(32, (3,3), activation = 'relu'))
classificator.add(BatchNormalization())
classificator.add(MaxPooling2D(pool_size=(2,2)))
classificator.add(Flatten())

#min = 90
classificator.add(Dense(units = 64, activation = 'relu'))
classificator.add(Dropout(0.2))
classificator.add(Dense(units = 64, activation = 'relu'))
classificator.add(Dropout(0.2))
classificator.add(Dense(units = 10, activation = 'softmax'))
classificator.compile(loss = 'categorical_crossentropy',
                      optimizer='adam', 
                      metrics = ['accuracy'])
classificator.fit(predictors_train, classes_train, batch_size=128, epochs=5, validation_data=(predictors_test, classes_test))

result = classificator.evaluate(predictors_test, classes_test)
