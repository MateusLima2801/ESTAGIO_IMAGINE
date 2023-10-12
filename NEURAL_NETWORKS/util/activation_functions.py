import numpy as np
# transfer 

def step(x):
    if (x >= 1):
        return 1
    return 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tahn(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    if x >= 0:
        return x
    return 0

def linear(x):
    return x

def softmax(x: list) -> list:
    ex = np.exp(x) # list
    return ex / ex.sum() # list/sum of the list

teste = step(-1)
teste = sigmoid(-0.358)
teste = tahn(-0.358)
teste = relu(0.358)
teste = linear(-0.358)
valores = [7.0, 2.0, 1.3]

print(softmax(valores))