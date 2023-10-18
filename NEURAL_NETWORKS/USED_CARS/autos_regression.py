import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('USED_CARS/autos.csv', encoding = 'ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('postalCode', axis = 1)

name_count = base['name'].value_counts()
base = base.drop('name', axis = 1)

seller_count = base['seller'].value_counts()
base = base.drop('seller', axis = 1)

count = base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)

s1 = base.loc[base.price <= 10]
base = base[base.price > 10]
s2 = base.loc[base.price > 350000]
base = base[base.price<350000]

# s3 = base.loc[pd.isnull(base['vehicleType'])]
# count = base['vehicleType'].value_counts() #limousine

# s4 = base.loc[pd.isnull(base['gearbox'])]
# count = base['gearbox'].value_counts() #manuell

# s4 = base.loc[pd.isnull(base['model'])]
# count = base['model'].value_counts() #golf

# s4 = base.loc[pd.isnull(base['fuelType'])]
# count = base['fuelType'].value_counts() #benzin

# s4 = base.loc[pd.isnull(base['notRepairedDamage'])]
# count = base['notRepairedDamage'].value_counts() #nein

values = {'vehicleType': 'limousine',
          'gearbox': 'manuell',
          'model': 'golf',
          'notRepairedDamage': 'nein',
          'fuelType': 'benzin'}

base = base.fillna(value = values)
predictors = base.iloc[:, 1:13].values
real_price = base.iloc[:, 0].values

print(predictors[0])

# encoder to number
encoder_predictors = LabelEncoder()
idxes = [0, 1, 3, 5, 8, 9, 10]
for i in idxes:
    predictors[:, i] = encoder_predictors.fit_transform(predictors[:, i])

# print(predictors[0])

# encoder to one hot encoder
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), idxes)],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
predictors = ct.fit_transform(predictors).toarray()
# print(len(predictors[0]))

#neural network
regressor = Sequential()
regressor.add(Dense(units=158, activation='relu', input_dim=316))
regressor.add(Dense(units=158, activation='relu'))
regressor.add(Dense(units=1, activation='linear'))
regressor.compile(loss='squared_hinge', optimizer='adam', 
                  metrics=['squared_hinge'])
regressor.fit(predictors, real_price, batch_size=500, epochs=10)

predictions = regressor.predict(predictors)