from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.util import flatten
from optimizers import adam
import matplotlib.pyplot as plt

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification. params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix. returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs

if __name__ == '__main__':
    
    # Model and training parameters
    layer_sizes = [1,10,10,1]
    param_scale, step_size = 1.0, 0.1
    inputs = np.array([[-1.0],[-0.875],[-0.75],[-0.625],[-0.5],[0.5],[0.625],[0.75],[0.875],[1.0]])
    targets = np.array([[ 1.17],[ 0.92],[ 0.64],[ 0.30],[-0.23],[0.86],[1.07],[0.74],[0.34],[-0.10]])

    # Randomly initialize the neural net parameters
    init_params = init_random_params(param_scale, layer_sizes)

    # Define training objective, equivalent to the log_posterior of the distribution
    def objective(params, iter):
        return np.sum((neural_net_predict(params, inputs) - targets)**2)

    # Use autograd to obtain the gradient of the objective function
    objective_grad = grad(objective)

    def print_function(params, iter, gradient):
    	""" Print the error at every iteration """
    	if iter % 10 == 0:
    		print("Total Error:", np.sum(np.absolute(neural_net_predict(params, inputs) - targets)))

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=500, callback=print_function)

