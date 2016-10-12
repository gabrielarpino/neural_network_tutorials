import autograd.numpy as np
from autograd import grad

import matplotlib.pyplot as plt

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

# Set up figure.
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show()

# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])
count = 0
weights_list = []

# Define a function that returns gradients of training loss using autograd.
training_gradient_fun = grad(training_loss)

# Optimize weights using gradient descent.
weights = np.array([0.0, 0.0, 0.0])
print "Initial loss:", training_loss(weights)
for i in xrange(1000):
    weights -= training_gradient_fun(weights) * 0.9
    print "Updated loss:", training_loss(weights)
    weights_list.append(weights[0])

    # Plot data and functions.
    count = count + 1
    plot_inputs = np.linspace(0, count+1, num = 1)

    ax.plot(weights_list)
    plt.draw()
    plt.pause(1.0/60.0)

print  "Trained loss:", training_loss(weights)