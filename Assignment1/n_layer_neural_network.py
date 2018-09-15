import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

f__author__ = 'Wilfredo J. Molina'


def generate_data(quantity, seed, noise):
    np.random.seed(seed)
    x, y = datasets.make_moons(quantity, noise = noise)
    return x, y


def plot_decision_boundary(predictor, x, y):
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = .01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = predictor(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap = plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Spectral)
    plt.show()


class DeepNeuralNetwork(object):
    def __init__(self, dimensions, activation_type, regularization, random_seed):
        np.random.seed(random_seed)

        self.dimensions = np.array(dimensions)
        self.activation_type = activation_type
        self.regularization = regularization

        self.w = dict([(i + 1, 2 * np.random.random([self.dimensions[i], self.dimensions[i - 1]]) - 1) for i in range(1, len(self.dimensions))])
        self.b = dict([(i + 1, 2 * np.random.random([self.dimensions[i], 1]) - 1) for i in range(1, len(self.dimensions))])

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

    def activation_derivative(self, x):
        if self.activation_type == 'tanh':
            return np.cosh(x) ** - 2
        elif self.activation_type == 'sigmoid':
            y = self.activation(x)
            return y * (1 - y)
        elif self.activation_type == 'relu':
            return x > 0
        elif self.activation_type == 'gaussian':
            return - 2 * x * self.activation(x)
        elif self.activation_type == 'sinusoid':
            return np.cos(x)
        elif self.activation_type == 'softplus':
            return (1 + np.exp(- x)) ** - 1

    @staticmethod
    def soft_max(x):
        return np.exp(x) / np.sum(np.exp(x), axis = 0)

    def back_propagation(self, a1, y):
        self.a[1] = np.array(a1)

        d_b = dict()
        d_w = dict()
        delta = dict()

        delta[len(self.dimensions)] = self.a[len(self.dimensions)] - y

        for l in range(len(self.dimensions) - 1, 1, - 1):
            delta[l] = self.w[l + 1].T @ delta[l + 1] * self.activation_derivative(self.z[l])

        for l in range(2, len(self.dimensions) + 1):
            d_b[l] = np.sum(delta[l], axis = 1, keepdims = True) / len(a1[0])
            d_w[l] = np.einsum('ik, jk -> ij', delta[l], self.a[l - 1]) / len(a1[0])

        return d_w, d_b

    def train(self, a1, y, train_rate, passes, print_loss, print_rate):
        for i in range(0, passes):
            self.feed_forward(a1)

            d_w, d_b = self.back_propagation(a1, y)

            for j in range(2, len(self.dimensions) + 1):
                d_w[j] += self.regularization * self.w[j]
                self.b[j] -= train_rate * d_b[j]
                self.w[j] -= train_rate * d_w[j]

            if print_loss and i % print_rate == 0:
                print(str(i).rjust(len(str(passes))) + ': ' + '{0:.5f}'.format(self.calculate_loss(a1, y)))

    def calculate_loss(self, a1, y):
        return - np.sum(y * np.log(self.a[len(self.dimensions)])) / len(a1[0])

    def predict(self, a1):
        self.feed_forward(a1)
        return np.argmax(self.a[len(self.dimensions)], axis = 0)


def main():
    # Make Moons

    # a1, y = generate_data(quantity = 1000, seed = 0, noise = .1)
    #
    # a1 = a1.T
    # y = np.array([[i == 0, i == 1] for i in y]).T
    #
    # network = DeepNeuralNetwork([2, 10, 2], 'relu', 0, 0)
    # network.train(a1, y, .1, 10000, True, 100)
    #
    # plot_decision_boundary(lambda x: network.predict(x.T), a1.T, y[1])

    # Make Circles

    a1, y = datasets.make_circles(n_samples = 1000, shuffle = True, noise = .05, random_state = None, factor = 0.8)
    a1 = a1.T
    y = np.array([[i == 0, i == 1] for i in y]).T

    network = DeepNeuralNetwork([2, 15, 2], 'sinusoid', 0, 0)
    network.train(a1, y, .1, 10000, True, 100)

    x_min, x_max = a1.T[:, 0].min() - .5, a1.T[:, 0].max() + .5
    y_min, y_max = a1.T[:, 1].min() - .5, a1.T[:, 1].max() + .5
    h = .01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = network.predict(np.c_[xx.ravel(), yy.ravel()].T)
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap = plt.cm.Spectral)
    plt.scatter(a1.T[:, 0], a1.T[:, 1], c = y[1], cmap = plt.cm.Spectral)
    plt.show()


if __name__ == '__main__':
    main()
