from Functions import *

# 1 Visualizing a CNN with CIFAR10

# 1.1 CIFAR10 Dataset

x_train, y_train, x_test, y_test = load_modified_cifar10()

# 1.2 Train LeNet5 on CIFAR10

from itertools import product
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterSampler, train_test_split

graph = tf.Graph()
x_init = tf.contrib.layers.xavier_initializer()

with graph.as_default():
    x = tf.placeholder(tf.float32, shape = [None, 28, 28])
    y = tf.placeholder(tf.float32, shape = [None, 10])

    x_one_ch = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight((5, 5, 1, 32))
    b_conv1 = tf.constant(0.1, shape = (32,))
    h_conv1 = tf.nn.relu(conv2d(x_one_ch, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight((5, 5, 32, 64))
    b_conv2 = tf.constant(0.1, shape = (64,))
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight((7 * 7 * 64, 1024))
    b_fc1 = tf.constant(0.1, shape = (1024,))
    h_pool2_flat = tf.reshape(h_pool2, (-1, 7 * 7 * 64))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight((1024, 10))
    b_fc2 = tf.constant(0.1, shape = (10,))
    y_out = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_out))
    correct_pred = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

n_iter = 10
param_sch_epochs = 10
batch_size = 100
full_train_epochs = 50
full_train_monitor_iters = 100

training_params = ParameterSampler({ 'optimizer': [tf.train.AdamOptimizer(r, b, epsilon = eps) for r, b, eps in product([1e-4, 3e-4, 1e-3], [0.8, 0.9, 0.95], [1e-8, 1e-6, 1e-4])], 'keep_prob': [0.3, 0.4, 0.5, 0.6, 0.7] }, n_iter = n_iter)
x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train, test_size = 0.1)

param_search_results = []
for i, param in enumerate(training_params):
    print('Evaluating Parameter Set: %d' % (i + 1))
    eval_train, eval_test, sess = train_network(x_train_train, y_train_train, x_train_test, y_train_test, graph, x, y, param['optimizer'], param_sch_epochs, batch_size, cross_entropy, acc, train_feed_dict = { keep_prob: param['keep_prob'] },
                                                eval_feed_dict = { keep_prob: 1.0 })
    sess.close()
    param_search_results.append((param, (eval_train, eval_test)))

best_params = max(param_search_results, key = lambda x: x[1][1])[0]
opt, p = best_params['optimizer'], best_params['keep_prob']

print('Training with Best Parameters: (%g, %g, %g, %g)' % (opt._lr, opt._beta1, opt._epsilon, p))
eval_train, eval_test, best_sess, monitor_train, monitor_test = train_network(x_train, y_train, x_test, y_test, graph, x, y, opt, full_train_epochs, batch_size, cross_entropy, acc, train_feed_dict = { keep_prob: p }, eval_feed_dict = { keep_prob: 1.0 },
                                                                              monitor_vars = (acc, cross_entropy), monitor_iters = full_train_monitor_iters)

print('Train Accuracy : %.4f' % eval_train)
print('Test Accuracy : %.4f' % eval_test)

iter, train_vals = zip(*monitor_train)
train_accs, train_losses = zip(*train_vals)

_, test_vals = zip(*monitor_test)
test_accs, test_losses = zip(*test_vals)

plt.subplot(1, 2, 1)
plt.plot(np.array(iter) / x_train.shape[0], train_accs, label = 'Train')
plt.plot(np.array(iter) / x_train.shape[0], test_accs, label = 'Test')
plt.title('Epoch vs Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1.2)

plt.subplot(1, 2, 2)
plt.plot(np.array(iter) / x_train.shape[0], train_losses, label = 'Train')
plt.plot(np.array(iter) / x_train.shape[0], test_losses, label = 'Test')
plt.title('Epoch vs Cross Entropy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.ylim(0, 2.5)

plt.tight_layout()
plt.show()

# 1.3 Visualize the Trained Network

w1 = best_sess.run(W_conv1)
w1_std = np.std(w1.reshape((-1, w1.shape[-1])), axis = 0)
best_indices = list(np.argsort(w1_std)[:: -1][: 4])

vmax = np.abs(w1).max()
vmin = -vmax

for i, j in product(range(4), range(8)):
    k = i * 8 + j
    plt.subplot(4, 8, k + 1)
    plt.pcolormesh(np.flipud(w1[:, :, 0, k]), vmin = vmin, vmax = vmax, cmap = 'gray')
    plt.axis('off')

plt.show()
gridspec = plt.GridSpec(4, 8)

for i1, j1 in product(range(2), range(2)):
    k = i1 * 2 + j1
    plt.subplot(gridspec[i1 * 2: i1 * 2 + 2, j1 * 4: j1 * 4 + 2])
    img = x_test[200 * k + 10, :, :]
    plt.pcolormesh(np.flipud(img), vmin = 0, vmax = 1, cmap = 'gray')
    plt.axis('off')
    h1 = best_sess.run(h_pool1, feed_dict = { x: img[None, :, :] })

    for i2, j2 in product(range(2), range(2)):
        k2 = i2 * 2 + j2
        plt.subplot(gridspec[i1 * 2 + i2, j1 * 4 + 2 + j2])
        h = h1[0, :, :, best_indices[k2]]
        plt.pcolormesh(np.flipud(h), cmap = 'gray')
        plt.axis('off')

plt.show()
