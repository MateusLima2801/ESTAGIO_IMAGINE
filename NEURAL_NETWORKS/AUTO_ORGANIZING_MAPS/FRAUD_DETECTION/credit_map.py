from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot 

base = pd.read_csv('/home/mateus/Desktop/MACHINE_LEARNING/NEURAL_NETWORKS/AUTO_ORGANIZING_MAPS/FRAUD_DETECTION/credit_data.csv')
base = base.dropna()
mean_age = base.age.mean()
base.loc[base.age<0, 'age'] = mean_age
x = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

norm = MinMaxScaler(feature_range=(0,1))
x  = norm.fit_transform(x)

# len(x)=1997
# 5*sqrt(1997) ~ 223
# 15x15 map
som =  MiniSom(x=15, y=15, input_len = 4, random_seed = 0)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, el in enumerate(x):
    w = som.winner(el)
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgecolor=colors[y[i]],
         markeredgewidth=2)

mapping = som.win_map(x)
suspects = np.concatenate((mapping[(4,5)], mapping[(6,13)]), axis=0)
suspects = norm.inverse_transform(suspects)

classes = []
for i in range(len(base)):
    for j in range(len(suspects)):
        if base.iloc[i,0] == int(round(suspects[j,0])):
            classes.append(base.iloc[i,4])
classes = np.asarray(classes)

final_suspects = np.column_stack((suspects, classes))

final_suspects = final_suspects[final_suspects[:,4].argsort()]