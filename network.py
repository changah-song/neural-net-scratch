from loss import mse, mse_prime

class Network:
  def __init__(self):
    self.layers = []
    self.loss = mse
    self.loss_prime = mse_prime

  # add new layer
  def add(self, layer): self.layers.append(layer)
  
  # train the model with data
  def fit(self, x_train, y_train, epochs, learning_rate):
    for i in range(epochs):
      err = 0
      for j in range(len(x_train)):
        output = x_train[j]
        # forward propagation
        for layer in self.layers:
          output = layer.forward_propagation(output)
          # output is y_pred after end of forward propagation

        # error calculation
        err += self.loss(y_train[j], output) # error 
        dEdY = self.loss_prime(y_train[j], output) # error prime AKA dEdY

        # backward propagation
        for layer in reversed(self.layers):
          dEdY = layer.backward_propagation(dEdY, learning_rate)
      
      # print curren epoch and average error
      err /= len(x_train)
      print('epoch %d/%d   error=%f' % (i+1, epochs, err))
    
  # predicted output of model given input
  def predict(self, input_data):
    result = []
    # forward propagate once
    for i in range(len(input_data)):
      output = input_data[i]
      for layer in self.layers:
        output = layer.forward_propagation(output)
      result.append(output)
    return result