import time
from n_layer_neural_network import DeepNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

f__author__ = 'Wilfredo J. Molina'


class NeuralNetwork(DeepNeuralNetwork):
    def __init__(self, input_dim = 2, hidden_dim = 3, output_dim = 2, activation_type = 'tanh', regularization = 0, random_seed = None):
        super().__init__([input_dim, hidden_dim, output_dim], activation_type, regularization, random_seed)


def main():
    I = [[0.00, '00'], [.025, '01'], [.075, '02']]
    J = [['ReLU', 'relu', '00'], ['sigmoid', 'sigmoid', '01'], ['hyperbolic tangent', 'tanh', '02']]
    K = [['00', lambda x: datasets.make_moons(n_samples = 1000, shuffle = True, noise = x, random_state = None)], ['01', lambda x: datasets.make_circles(n_samples = 1000, shuffle = True, noise = x, random_state = None, factor = 0.75)]]
    L = [[10, '00'], [20, '01'], [30, '02']]

    for l in L:
        for k in K:
            for j in J:
                for i in I:
                    # Data

                    a1, y = k[1](i[0])

                    # Pre-Processing

                    a1 = a1.T
                    y = np.array([[i == 0, i == 1] for i in y]).T

                    # Network

                    network = NeuralNetwork(input_dim = 2, hidden_dim = l[0], output_dim = 2, activation_type = j[1], regularization = 0, random_seed = None)
                    network.train(a1, y, .1, 100000, True, .2)

                    # Display

                    x_min, x_max = a1.T[:, 0].min() - .5, a1.T[:, 0].max() + .5
                    y_min, y_max = a1.T[:, 1].min() - .5, a1.T[:, 1].max() + .5
                    h = .01
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                    z = network.predict(np.c_[xx.ravel(), yy.ravel()].T)
                    z = z.reshape(xx.shape)
                    plt.figure(figsize = (5, 5))
                    plt.axis('off')
                    plt.title(j[0] + ', ' + str(l[0]) + ', ' + '{0:.3f}'.format(i[0]))
                    plt.contourf(xx, yy, z, cmap = plt.cm.Spectral)
                    plt.scatter(a1.T[:, 0], a1.T[:, 1], c = y[1], cmap = plt.cm.Spectral)
                    plt.savefig('./images/' + i[1] + j[2] + k[0] + l[1] + '.png')
                    # plt.show(block = False)
    # plt.show()


if __name__ == '__main__':
    main()
