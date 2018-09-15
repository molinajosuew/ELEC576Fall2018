import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

f__author__ = 'Wilfredo J. Molina'


def generate_data(quantity = 20, seed = 0, noise = .02):
    np.random.seed(seed)
    x, y = datasets.make_moons(quantity, noise = noise)
    return x, y


def plot_decision_boundary(predict, x, y):
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = .01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap = plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c = y, cmap = plt.cm.Spectral)
    plt.show()


class ThreeLayerNetwork:
    def __init__(self, input_dim = 2, hidden_dim = 3, output_dim = 2, activation_type = 'tanh', regularization = .01, seed = 0):
        np.random.seed(seed)

        self.input_dimension = input_dim
        self.hidden_dimension = hidden_dim
        self.output_dimension = output_dim

        self.activation_type = activation_type

        self.regularization = regularization

        self.W2 = 2 * np.random.random([self.hidden_dimension, self.input_dimension]) - 1
        self.b2 = 2 * np.random.random([self.hidden_dimension, 1]) - 1
        self.W3 = 2 * np.random.random([self.output_dimension, self.hidden_dimension]) - 1
        self.b3 = 2 * np.random.random([self.output_dimension, 1]) - 1

        self.z2 = None
        self.a2 = None
        self.z3 = None
        self.a3 = None

    def activation(self, x):
        if self.activation_type == 'tanh':
            return np.tanh(x)
        elif self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(- x))
        elif self.activation_type == 'relu':
            return x * (x > 0)

    def activation_derivative(self, x):
        if self.activation_type == 'tanh':
            return np.cosh(x) ** - 2
        elif self.activation_type == 'sigmoid':
            y = self.activation(x)
            return y * (1 - y)
        elif self.activation_type == 'relu':
            return x > 0

    def feed_forward(self, a1):
        self.z2 = self.W2 @ a1 + self.b2
        self.a2 = self.activation(self.z2)

        self.z3 = self.W3 @ self.a2 + self.b3
        self.a3 = np.exp(self.z3) / np.sum(np.exp(self.z3), axis = 0)

    def calculate_loss(self, a1, y):
        return - np.sum(y * np.log(self.a3)) / len(a1)

    def predict(self, a1):
        self.feed_forward(a1)
        return np.argmax(self.a3, axis = 0)

    def back_propagation(self, a1, y):
        delta3 = self.a3 - y
        delta2 = self.W3.T @ delta3 * self.activation_derivative(self.z2)

        d_b2 = np.sum(delta2, axis = 1, keepdims = True) / len(a1)
        d_w2 = np.einsum('ik, jk -> ij', delta2, a1) / len(a1)
        d_b3 = np.sum(delta3, axis = 1, keepdims = True) / len(a1)
        d_w3 = np.einsum('ik, jk -> ij', delta3, self.a2) / len(a1)

        return d_b2, d_w2, d_b3, d_w3

    def train(self, a1, y, train_rate = .01, num_passes = 10000, print_loss = False, print_rate = 100):
        for i in range(0, num_passes):
            self.feed_forward(a1)
            d_b2, d_w2, d_b3, d_w3 = self.back_propagation(a1, y)

            d_w3 += self.regularization * self.W3
            d_w2 += self.regularization * self.W2

            self.b2 -= train_rate * d_b2
            self.W2 -= train_rate * d_w2
            self.b3 -= train_rate * d_b3
            self.W3 -= train_rate * d_w3

            if print_loss and i % print_rate == 0:
                print(f'{i}: {self.calculate_loss(a1, y)}')

    def visualize_decision_boundary(self, a1, y):
        plot_decision_boundary(lambda x: self.predict(x.T), a1, y)


def main():
    a1, y = generate_data(quantity = 1000, seed = 0, noise = .01)

    a1 = a1.T
    y = np.array([[i == 0, i == 1] for i in y]).T

    network = ThreeLayerNetwork(input_dim = 2, hidden_dim = 3, output_dim = 2, activation_type = 'sigmoid', regularization = .0, seed = 0)

    network.train(a1, y, train_rate = .001, num_passes = 100000, print_loss = True, print_rate = 1000)

    network.visualize_decision_boundary(a1.T, y[1])


if __name__ == '__main__':
    main()
