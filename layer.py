import numpy as np

class Layer:
  def __init__(self, input_size, output_size, activation, activation_prime):
    self.input = None
    self.output = None
    # initialize weights and biases randomly
    self.weights = np.random.rand(input_size, output_size) - 0.5
    self.bias = np.random.rand(1, output_size) - 0.5
    self.activation = activation
    self.activation_prime = activation_prime
  
  def forward_propagation(self, input_data):
    # input (X)
    self.input = input_data
    # middle = WX + b ; save for back propagation
    self.middle = np.dot(self.input, self.weights) + self.bias
    # output (Y) = activation(WX + b)
    self.output = self.activation(self.middle)
    return self.output

  def backward_propagation(self, dEdY, learning_rate):
    # --- find gradients ---
    # Given Y = activation(WX + b), use CHAIN RULE to find partial derivatives:
      # dY/dW = activation'(WX + b) * X
      # dY/dX = activation'(WX + b) * W
      # dY/db = activation'(WX + b)
    
    # element-wise multiplication for dEdY and activation'(WX + b) as their dimension is equal
    # matrix multiplication for X and the result

    # dEdW = dEdY * dYdW = dEdY * activation'(WX + b) * X
    dEdW = np.dot(self.input.T, (self.activation_prime(self.middle) * dEdY))
    # dEdX = dEdY * dYdX = dEdY * activation'(WX + b) * W
    dEdX = np.dot((self.activation_prime(self.middle) * dEdY), self.weights.T)
    # dEdb = dEdY * dYdb = dEdY * activation'(WX + b)
    dEdb = (self.activation_prime(self.middle) * dEdY)

    # --- update parameters ---
    self.weights -= learning_rate * dEdW
    self.bias -= learning_rate * dEdb
  
    # return dEdX which can be thought of as the 
    ### new dEdY that will be passed to the previous layer ###
    return dEdX