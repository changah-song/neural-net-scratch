from neuralnetwork import NeuralNetwork
from layer import Layer
from activation import tanh, tanh_derivative

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
import pickle
batch1 = unpickle("cifar 10 dataset/data_batch_1")
batch1.keys()
# there are four keys in the dictionary: batch_label, labels (numbered 0~9 for each class), data, and filenames
# important ones are "labels" and "data"

# meta data tells us what classes the numbers correspond to
# {0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck}

from keras.utils import np_utils

train_x = batch1[b'data'][:1000]
train_y = batch1[b'labels'][:1000] 

test_x = batch1[b'data'][1000:5000]
test_y = batch1[b'labels'][1000:5000] 

# reshape and normalize input data
train_x = train_x.reshape(train_x.shape[0], 1, 3072)
train_x = train_x.astype('float32')
train_x /= 255
train_y = np_utils.to_categorical(train_y)

test_x = test_x.reshape(test_x.shape[0], 1, 3072)
test_x = test_x.astype('float32')
test_x /= 255
test_y = np_utils.to_categorical(test_y)

model = NeuralNetwork()
model.add(Layer(3072, 2000, tanh, tanh_derivative))
model.add(Layer(2000, 1000, tanh, tanh_derivative))
model.add(Layer(1000, 300, tanh, tanh_derivative))
model.add(Layer(300, 10, tanh, tanh_derivative))
model.compile(train_x, train_y, epochs=5, learning_rate=0.01)