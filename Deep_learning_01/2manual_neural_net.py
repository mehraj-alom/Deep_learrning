import numpy as np 
np.random.seed(4)



class neu_net:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.weight_input_to_hidden = np.random.randn(n_inputs, n_hidden)
        self.bias_hidden = np.zeros((1,n_hidden))

        self.weight_hidden_to_output = np.random.randn(n_hidden, n_outputs)
        self.bias_output = np.zeros((1,n_outputs))
        
        self.learning_rate = 0.1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    def forward(self,x):
        self.input_layer = x
        
        self.hidden_input = np.dot(self.input_layer,self.weight_input_to_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output,self.weight_hidden_to_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output
    
    def backward(self, expected_output):
        output_error = expected_output - self.final_output
        output_delta = output_error * self.sigmoid_derivative(self.final_output)
        
        hidden_error = np.dot(output_delta,self.weight_hidden_to_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.weight_input_to_hidden += np.dot(self.input_layer.T, hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        
        self.weight_hidden_to_output += np.dot(self.hidden_input, output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        
        
    def train(self, inputs, expected_output, epochs):
        for epoch in range(epochs):
            output = self.forward(inputs)
            self.backward(expected_output)

            if epoch % 200 == 0:
                loss = np.mean(np.abs(expected_output - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
if __name__ == "__main__":
    # Example usage
    nn = neu_net(3, 4, 1)
    inputs = np.array([[0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1],
                       [1, 1, 1]])
    expected_output = np.array([[0], [1], [1], [0]])  # XOR problem
    nn.train(inputs, expected_output, epochs=10000)
    print("Final output after training:")
    print(nn.forward(inputs))
    # Expected output: [[0], [1], [1], [0]]
    
    
 