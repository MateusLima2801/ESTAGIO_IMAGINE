import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

#mnist has 28x28 images then 784 pixels

(x_train, y_train), (x_test, y_test) = mnist.load_data()
predictors_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
predictors_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

classes_train = np_utils.to_categorical(y_train, 10)
classes_test = np_utils.to_categorical(y_test, 10)

train_generator = ImageDataGenerator(rotation_range=7, horizontal_flip=True, shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)
test_generator = ImageDataGenerator() # default: do nothing

base_train = train_generator.flow(predictors_train, classes_train, batch_size = 128)
base_test = train_generator.flow(predictors_test, classes_test, batch_size = 128)


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

classificator.fit_generator(base_train, steps_per_epoch=60000/128, epochs=5, validation_data=base_test, validation_steps=10000/128)
