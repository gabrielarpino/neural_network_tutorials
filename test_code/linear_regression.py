import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

# Perform linear regression of the form y = mx + b

def predictions(weights, inputs):
	return np.dot(inputs,weights)

def cost_function(weights):
	preds = predictions(weights, inputs)
	print "predictions", preds
	cost = (preds - targets)**2
	print "cost", cost
	return -np.sum(np.log(cost))


# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([0.45, 0.33, -0.89, -0.11])

training_gradient_fun = grad(cost_function)

# Optimize weights using gradient descent.
weights = np.array([0.0, 0.0, 0.0])
b = np.array([0.0, 0.0, 0.0]).T
print "Initial loss:", cost_function(weights)

# Set up figure.
fig = plt.figure(figsize=(12, 8), facecolor='white')
ax = fig.add_subplot(111, frameon=False)
plt.ion()
plt.show(block=False)

for i in xrange(2000):
    weights += training_gradient_fun(weights) * 0.1
    print "training_gradient_fun(weights)", training_gradient_fun(weights)
    print "loss:", cost_function(weights)

    plot_inputs = np.linspace(1, 4, num=4)

    # Plot functions
    plt.cla()
    ax.plot(plot_inputs, targets, 'bo')
    ax.plot(plot_inputs, predictions(weights,inputs))
    plt.draw()
    plt.pause(1.0/60.0)

print  "Trained loss:", cost_function(weights)