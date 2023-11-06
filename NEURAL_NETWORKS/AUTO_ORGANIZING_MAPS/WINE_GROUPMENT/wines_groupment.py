from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

base = pd.read_csv('/home/mateus/Desktop/MACHINE_LEARNING/NEURAL_NETWORKS/AUTO_ORGANIZING_MAPS/WINE_GROUPMENT/wines.csv')
x = base.iloc[:,1:14].values
y = base.iloc[:,0].values

norm = MinMaxScaler(feature_range=(0,1))
x = norm.fit_transform(x)

# size ~ 5*sqrt(N)
# N = number of registers
# in that case, we have 178 registers then the size should be 65,55 cells
# approximating we can have a map 8x8 with 64 cells

som = MiniSom(x=8, y=8, input_len=13, sigma=1.0, learning_rate=0.5, random_seed=2)
som.random_weights_init(x)
som.train_random(data = x, num_iteration=100)
q = som.activation_response(x)

pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

#y  = y - 1 #fixing offset

for i, el in enumerate(x):
    print(i)
    print(el)
    w = som.winner(el)
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgecolor=colors[y[i]],
         markeredgewidth=2)