from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

OFFSET = 50
base = pd.read_csv('/home/mateus/Desktop/MACHINE_LEARNING/NEURAL_NETWORKS/EXCHANGE_RECORRENT/data/petr4_treinamento.csv')
base = base.dropna()
base_train = base.iloc[:,1:2].values

norm = MinMaxScaler(feature_range=(0,1))
base_train_norm = norm.fit_transform(base_train)

#prediction over open value

predictors = []
real_price = []
for i in range(OFFSET,len(base_train_norm)):
    predictors.append(base_train_norm[i-OFFSET:i,0])
    real_price.append(base_train_norm[i,0])
predictors, real_price = np.array(predictors), np.array(real_price)

predictors = np.reshape(predictors, (predictors.shape[0], predictors.shape[1], 1))


regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape = (predictors.shape[1], 1)))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))
# regressor.add(LSTM(units=50, return_sequences=True))
# regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))
regressor.add(Dense(units = 1, activation='linear'))
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])
regressor.fit(predictors, real_price, epochs = 10, batch_size=32)

base_test = pd.read_csv('/home/mateus/Desktop/MACHINE_LEARNING/NEURAL_NETWORKS/EXCHANGE_RECORRENT/data/petr4_teste.csv')
real_price_test = base_test.iloc[:,1:2].values
base_full = pd.concat((base['Open'], base_test['Open']), axis=0)
inputs = base_full[len(base_full)-len(base_test)-OFFSET:].values
inputs = inputs.reshape(-1,1)
inputs = norm.transform(inputs)

x_test = []
for i in range(OFFSET, len(inputs)):
    x_test.append(inputs[i-90:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = regressor.predict(x_test)
predictions = norm.inverse_transform(predictions)
print(predictions.mean(), real_price_test.mean())