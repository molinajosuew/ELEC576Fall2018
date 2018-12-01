import numpy as np


class DeepNeuralNetwork(object):
    def __init__(self, architecture):
        np.random.seed(None)

        self.architecture = architecture
        self.activation_type = 'tanh'
        self.regularization = 0

        self.A = - 20
        self.B = 20

        self.W = dict([(i + 1, (self.B - self.A) * np.random.random([self.architecture[i], self.architecture[i - 1]]) + self.A) for i in range(1, len(self.architecture))])
        self.b = dict([(i + 1, (self.B - self.A) * np.random.random([self.architecture[i], 1]) + self.A) for i in range(1, len(self.architecture))])

        self.a = dict()
        self.z = dict()

    def feed_forward(self, a1):
        self.a[1] = np.array(a1)
        self.z[2] = self.W[2] @ self.a[1] + self.b[2]

        for i in range(2, len(self.architecture)):
            self.a[i] = np.tanh(self.z[i])
            self.z[i + 1] = self.W[i + 1] @ self.a[i] + self.b[i + 1]

        self.a[len(self.architecture)] = np.exp(self.z[len(self.architecture)]) / np.sum(np.exp(self.z[len(self.architecture)]), axis = 0)

    def predict(self, a_1):
        self.feed_forward(a_1)
        return np.argmax(self.a[len(self.architecture)], axis = 0)[0]
