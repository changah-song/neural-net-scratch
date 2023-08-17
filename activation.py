import numpy as np

def sigmoid(x):
  return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1-np.tanh(x)**2