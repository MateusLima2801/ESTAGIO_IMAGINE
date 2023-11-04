from keras.models import model_from_json
import numpy as np

file = open(".\\NEURAL_NETWORKS\\BREAST_CANCER\\net_breast.json", "r")
net_structure = file.read()
file.close()

net = model_from_json(net_structure)
net.load_weights(".\\NEURAL_NETWORKS\\BREAST_CANCER\\net_breast.h5")

new  = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])
prediction = net.predict(new)
prediction = (prediction > 0.5)
print(prediction)