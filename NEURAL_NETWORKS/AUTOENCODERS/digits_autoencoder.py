import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import  Input, Dense

(predictors_train, _) ,  (predictors_test, _) = mnist.load_data()
predictors_train = predictors_train.astype('float32')/255
predictors_test = predictors_test.astype('float32')/255

predictors_train = predictors_train.reshape( (len(predictors_train), np.prod(predictors_train.shape[1:]) ))
predictors_test = predictors_test.reshape( (len(predictors_test), np.prod(predictors_test.shape[1:]) ))

#784-32-784
ratio = 784/32
#coding and decoding
#autoencoder training recquires more epochs
autoencoder = Sequential()
autoencoder.add(Dense(units=32, activation='relu', input_dim = 784))
autoencoder.add(Dense(units=784, activation='sigmoid'))
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(predictors_train, predictors_train, epochs=20, batch_size=256, 
                validation_data=(predictors_test,predictors_test))

original_dim = Input(shape=(784,))
encoder_layer = autoencoder.layers[0]
encoder = Model(original_dim, encoder_layer(original_dim))
encoder.summary()

encoded_images = encoder.predict(predictors_test)
decoded_images = autoencoder.predict(predictors_test)

imgs_number = 10
test_images = np.random.randint(predictors_test.shape[0], size = imgs_number)
plt.figure(figsize=(28,28))
for i, idx in enumerate(test_images):
    #original
    axis = plt.subplot(10,10,i+1)
    plt.imshow(predictors_test[idx].reshape(28,28))
    plt.xticks()
    plt.yticks()

    #coded image
    axis = plt.subplot(10,10,i+1+imgs_number)
    plt.imshow(encoded_images[idx].reshape(8,4))
    plt.xticks()
    plt.yticks()

    #decoded image
    axis = plt.subplot(10,10,i+1+imgs_number*2)
    plt.imshow(decoded_images[idx].reshape(28,28))
    plt.xticks()
    plt.yticks()