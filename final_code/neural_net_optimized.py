from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
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
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return np.tanh(outputs)

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik

def accuracy(params, inputs, targets):
    target_mean = np.mean(targets)
    print(neural_net_predict(params, inputs))
    multi = neural_net_predict(params, inputs)*targets
    multi_mean = np.mean(multi)
    return (np.absolute((multi_mean - target_mean)))

if __name__ == '__main__':
    # Model parameters
    layer_sizes = [3,4,4,1]
    L2_reg = 0.09

    # Training parameters
    param_scale = 0.1
    step_size = 0.001

    #Define the input arrays x and the desired output array y
    inputs = np.array([[1,1,-1]
        ,[1,-1,-1]
        ,[1,1,1]
        ,[-1,1,1]])
    targets = np.array([[-1,-1,1,1]]).T

    # Randomly initialize the neural net parameters
    init_params = init_random_params(param_scale, layer_sizes)

    # Define training objective
    def objective(params, iter):
        return -log_posterior(params, inputs, targets, L2_reg)

    # Use autograd to obtain the gradient of the objective function
    objective_grad = grad(objective)

    print("Accuracy:")
    def print_function(params, iter, gradient):
            # Print the accuracy once every 100 steps
            if (iter%100 == 0):
                print(accuracy(params,inputs,targets))

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=1000, callback=print_function)

