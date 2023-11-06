from rbm import RBM
import numpy as np

rbm = RBM(num_visible=6, num_hidden=2)
base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])

rbm.train(base, max_epochs=4999)

user1 = np.array([[1,1,0,1,0,0]])
user2 = np.array([[0,0,0,1,0,1]])

rbm.run_visible(user1)
rbm.run_visible(user2)

hidden = np.array([[0,1]])
recommendation = rbm.run_hidden(hidden)

hidden2 = np.array([[1,0]])
recommendation2 = rbm.run_hidden(hidden2)
