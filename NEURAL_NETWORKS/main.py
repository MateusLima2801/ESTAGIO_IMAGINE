from network import Network
from numpy import array
def main():
    n = Network([2,2,1])
    training_data= [(array([0,0]),array([0])),
         (array([0,1]),array([1])),
         (array([1,0]),array([1])),
         (array([1,1]),array([0]))]
    n.SGD(
        training_data=training_data
         , epochs=50,
         mini_batch_size=3,
         eta = 3)
    print(n.feedforward(array([0,0])))
    print(n.feedforward(array([0.5,  0.5])))
if __name__ == "__main__":
    main()