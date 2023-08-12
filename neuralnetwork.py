from loss import mse, mse_prime

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j].reshape(1, -1)
                # feed forward
                for layer in self.layers:
                    output = layer.forward(output)
                    # print("layer shapes", layer.input.shape, type(layer.input))
                # loss calculation
                err += mse(y_train[j], output)       
                # backpropagation
                error = mse_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
                    # print("gets fed into previous layer: \n", error.shape)
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            # forward propagation
            output = input_data[i].reshape(1, -1)
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result