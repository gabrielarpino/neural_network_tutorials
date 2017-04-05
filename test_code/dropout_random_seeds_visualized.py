from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import flatten
from optimizers import adam
import matplotlib.pyplot as plt
import random

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs, dropout, test_time = False):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
        if dropout:
            inputs *= np.random.binomial([np.ones_like(inputs)],(1-dropout_rate))[0]/(1-dropout_rate)
    return outputs

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = -np.sum((neural_net_predict(params, inputs, False) - targets)**2)
    return log_prior + log_lik

def log_posterior_dropout(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = -np.sum((neural_net_predict(params, inputs, True) - targets)**2)
    return log_prior + log_lik

def build_train_dataset(n_data=20, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets

def build_test_dataset(n_data=20, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(3, 6, num=n_data)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets

if __name__ == '__main__':

    # Create list of random seed inputs:
    seed_inputs = [10,20,30,40,50,60,70,80,90,100]

    # Model parameters
    layer_sizes = [1,10,10,1]
    L2_reg, dropout_rate = 0.0, 0.1

    # Training parameters
    param_scale = 1.0
    step_size = 0.1
    inputs, targets = build_train_dataset()
    testinputs, testtargets = build_test_dataset()

    # Randomly initialize the neural net parameters
    init_params = init_random_params(param_scale, layer_sizes)

    # Define training objective
    def objective(params, iter):
        return -log_posterior(params, inputs, targets, L2_reg)
    def objective_dropout(params, iter):
        return -log_posterior_dropout(params, inputs, targets, L2_reg)

    # Use autograd to obtain the gradient of the objective function
    objective_grad = grad(objective)
    objective_grad_dropout = grad(objective_dropout)

    testcostlist = []
    testcostlist_dropout = []
    traincostlist = []
    iterlist = []

    def print_function_dropout(params, iter, gradient):
        return

    def print_function(params, iter, gradient):
        use_dropout = False
        testcost = np.sum((neural_net_predict(params, testinputs, use_dropout) - testtargets)**2)
        testcostlist.append(testcost)

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=200, callback=print_function)

    optimized_params_list = []
    for i in range(len(seed_inputs)):
        random.seed(seed_inputs[i])
        optimized_params_list.append(adam(objective_grad_dropout, init_params, step_size=step_size, num_iters=200, callback=print_function_dropout))

    fig2, ax = plt.subplots()
    plt.cla()
    plt.title('Dropout Result Comparisons, dropout_rate:' + str(dropout_rate))
    ax.set_xlabel("Possible Inputs")
    ax.set_ylabel("Neural Network Outputs")
    plot_inputs = np.linspace(-8, 8, num=400)
    for i in range(len(seed_inputs)):
        # Plot data and functions.
        outputs = neural_net_predict(optimized_params_list[i], np.expand_dims(plot_inputs, 1), False, True)
        ax.plot(inputs, targets, 'bx')
        ax.plot(testinputs, testtargets, 'bo')
        ax.plot(plot_inputs, outputs, label='random.seed(%i)' % seed_inputs[i])
        ax.legend()
        ax.set_ylim([-2, 3])
        plt.draw()
        plt.pause(1.0)
    plt.show()
