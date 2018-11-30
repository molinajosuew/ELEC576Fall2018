import numpy as np


class DeepNeuralNetwork(object):
    def __init__(self, dimensions = np.array([1, 20, 20, 8]), activation_type = 'tanh', regularization = 0, random_seed = None):
        np.random.seed(random_seed)

        self.dimensions = np.array(dimensions)
        self.activation_type = activation_type
        self.regularization = regularization

        self.A = - 20
        self.B = 20

        self.w = dict([(i + 1, (self.B - self.A) * np.random.random([self.dimensions[i], self.dimensions[i - 1]]) + self.A) for i in range(1, len(self.dimensions))])
        self.b = dict([(i + 1, (self.B - self.A) * np.random.random([self.dimensions[i], 1]) + self.A) for i in range(1, len(self.dimensions))])

        self.a = dict()
        self.z = dict()

    def feed_forward(self, a1):
        self.a[1] = np.array(a1)
        self.z[2] = self.w[2] @ self.a[1] + self.b[2]

        for i in range(2, len(self.dimensions)):
            self.a[i] = self.activation(self.z[i])
            self.z[i + 1] = self.w[i + 1] @ self.a[i] + self.b[i + 1]

        self.a[len(self.dimensions)] = DeepNeuralNetwork.soft_max(self.z[len(self.dimensions)])

    def activation(self, x):
        if self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(- x))
        elif self.activation_type == 'relu':
            return (x > 0) * x
        elif self.activation_type == 'gaussian':
            return np.exp(- x ** 2)
        elif self.activation_type == 'sinusoid':
            return np.sin(x)
        elif self.activation_type == 'softplus':
            return np.log(1 + np.exp(x))
        elif self.activation_type == 'leakyrelu':
            y = x > 0
            return y * x + ~y * .01 * x
        elif self.activation_type == 'requ':
            return (x > 0) * x ** 2
        elif self.activation_type == 'step':
            return x > 0

    @staticmethod
    def soft_max(x):
        return np.exp(x) / np.sum(np.exp(x), axis = 0)

    def predict(self, a1):
        self.feed_forward(a1)
        return np.argmax(self.a[len(self.dimensions)], axis = 0)[0]
