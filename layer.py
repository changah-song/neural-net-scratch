import numpy as np

class Layer:
    def __init__(self, num_input, num_output, activation, activation_derivative):
        self.weight = np.random.rand(num_input, num_output)
        self.bias = np.random.rand(1, num_output)
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(np.dot(self.input, self.weight) + self.bias)
        return self.output

    def backward(self, dY, learning_rate):
        dX = np.dot(dY, self.weight.T)
        # print("input & dY shapes: ", self.input.T.shape, dY.shape)
        dw = np.dot(self.input.T, dY)
        db = dY
        
        self.weight -= learning_rate * dw
        self.bias -= learning_rate * dY
        temp = self.activation_derivative(self.input) * dX
        # print("dC/dY: \n", dY.shape)
        # print("output of backprop input dC/dY: \n", temp.shape)
        return self.activation_derivative(self.input) * dX