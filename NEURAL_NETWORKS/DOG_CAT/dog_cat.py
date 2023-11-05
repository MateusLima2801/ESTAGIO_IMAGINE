from keras.models  import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

classificator = Sequential()

classificator.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classificator.add(BatchNormalization())
classificator.add(MaxPooling2D(pool_size=(2,2)))

classificator.add(Flatten())

classificator.add(Dense(units=128, activation='relu'))
classificator.add(Dropout(0.2))
classificator.add(Dense(units=128, activation='relu'))
classificator.add(Dropout(0.2))
classificator.add(Dense(units = 1, activation='sigmoid'))
classificator.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics='accuracy')

gen_train = ImageDataGenerator(rescale = 1./255, rotation_range = 7, horizontal_flip=True, shear_range=0.2, height_shift_range=0.07,  zoom_range=0.2)
gen_test = ImageDataGenerator(rescale=1./255)

base_train = gen_train.flow_from_directory('/home/mateus/Desktop/MACHINE_LEARNING/NEURAL_NETWORKS/DOG_CAT/data/dataset/dataset/training_set', 
                                           target_size= (64,64),
                                           batch_size=32,
                                           class_mode='binary')
base_test = gen_test.flow_from_directory('/home/mateus/Desktop/MACHINE_LEARNING/NEURAL_NETWORKS/DOG_CAT/data/dataset/dataset/test_set', 
                                           target_size= (64,64),
                                           batch_size=32,
                                           class_mode='binary')

classificator.fit_generator(base_train, steps_per_epoch= 4000/32, epochs =5, validation_data=base_test, validation_steps=1000/32)

img_test = image.load_img('/home/mateus/Desktop/MACHINE_LEARNING/NEURAL_NETWORKS/DOG_CAT/data/dataset/dataset/test_set/cachorro/dog.3513.jpg', target_size=(64,64))
img_test = image.img_to_array(img_test)/255
img_test = np.expand_dims(img_test, axis=0)
prediction = classificator.predict(img_test)
prediction = (prediction > 0.5)
print(base_train.class_indices)
