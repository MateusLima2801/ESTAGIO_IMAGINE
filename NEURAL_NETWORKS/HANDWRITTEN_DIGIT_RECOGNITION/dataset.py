from torch import utils
from torchvision import datasets, transforms

# Transform PIL image into a tensor. The values are in the range [0, 1]
t = transforms.ToTensor()

# Load datasets for training and apply the given transformation.
mnist = datasets.MNIST(root='data', train=True, download=True, transform=t)

# Specify a data loader which returns 500 examples in each iteration.
n = 500
loader = utils.data.DataLoader(mnist, batch_size=n, shuffle=True)

for img, label in loader:
    print(label)