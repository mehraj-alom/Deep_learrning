import numpy as np 
np.random.seed(2)


class net:
    def __init__(self,n_inputs, n_neurons):
        self.weight = np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
    def sigmoid(self, x):   
        return 1 / (1 + np.exp(-x))
    def forward(self,inputs):
        self.output = self.sigmoid(0.10 * np.dot(inputs, self.weight)+ self.bias)

if __name__ == "__main__":
    X = np.array([[0.23, 0.45, 0.67],
              [0.12, 0.34, 0.56],
              [0.78, 0.89, 0.90],
              [0.11, 0.22, 0.33],
              [0.44, 0.55, 0.66]])
    layer1 = net(3,5)
    layer2 = net(5,2)
    
    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)