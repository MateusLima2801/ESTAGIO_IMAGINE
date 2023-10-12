import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

a1 = np.ones(5) #creates array
t = torch.from_numpy(a1) #creates tensor
a2 = t.numpy() #creates an array

t2 = torch.ones(5) #equivalent

print(t) #see on Debug mode

#Define a linear layer
linear_layer = nn.Linear(in_features=10, out_features=5)

weights = linear_layer.weight
bias = linear_layer.bias

#Define a linear layer without bias 
linear_layer = nn.Linear(in_features=10, out_features=5, bias = False)

#Define a ReLu llayer
relu_layer  = nn. ReLU()



# Define a MLP
inp = h = out = 1
mlp = nn.Sequential(
    nn.Linear(inp, h),
    nn.ReLU(), #activation after linear layer
    nn.Linear(h,out)
)

# define a class for 2 layers MLPs with ReLU non linearity
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP,self).__init__() #superclass of MLP
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
mlp = MLP(inp, h, out)

mse_loss = nn.MSELoss(reduction='sum')

prediction = torch.tensor([0.8, 0.3, 0.1])
target = torch.tensor([1., 0, 0])

loss = mse_loss(prediction, target)

print(loss)



