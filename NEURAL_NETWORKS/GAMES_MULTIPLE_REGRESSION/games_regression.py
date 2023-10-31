import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

base = pd.read_csv('GAMES_MULTIPLE_REGRESSION/games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)
base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

print(base['Name'].value_counts())

names = base.Name
base = base.drop('Name', axis = 1)

print(base) 

ipt_idxs = [0,1,2,3,7,8,9,10,11]
predictors = base.iloc[:,ipt_idxs].values
sales_na = base.iloc[:,4].values
sales_eu = base.iloc[:,5].values
sales_jp = base.iloc[:,6].values

encoder = LabelEncoder()
predictors[:, 0] = encoder.fit_transform(predictors[:, 0])
predictors[:, 2] = encoder.fit_transform(predictors[:, 2])
predictors[:, 3] = encoder.fit_transform(predictors[:, 3])
predictors[:, 8] = encoder.fit_transform(predictors[:, 8])

print(predictors[0])

categorical_idxs = [0,2,3,8]

# encoder to one hot encoder
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), categorical_idxs)],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
predictors = ct.fit_transform(predictors).toarray()
# print(len(predictors[0]), predictors)

L1 = Input(shape=[61,])
L2 = Dense(units = 32, activation = 'sigmoid')(L1)
L3 = Dense(units = 32, activation = 'sigmoid')(L2)
L4_1 = Dense(units = 1, activation = 'linear')(L3)
L4_2 = Dense(units = 1, activation = 'linear')(L3)
L4_3 = Dense(units = 1, activation = 'linear')(L3)

regressor = Model(inputs = L1, outputs = [L4_1, L4_2, L4_3])
regressor.compile(optimizer = 'adam', loss = 'mse')
regressor.fit(predictors, [sales_na, sales_eu, sales_jp], epochs = 1000, batch_size=100)