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

    # Set up figures.
    fig1 = plt.figure(figsize=(12, 8), facecolor='white')
    ax2 = fig1.add_subplot(212, frameon=False)
    ax = fig1.add_subplot(211, frameon=False)
    plt.ion()
    plt.show(block=False)

    testcostlist = []
    testcostlist_dropout = []
    traincostlist = []
    iterlist = []

    def print_function_dropout(params, iter, gradient):
        plot_inputs = np.linspace(-8, 8, num=400)
        use_dropout = False
        test_time = True
        outputs = neural_net_predict(params, np.expand_dims(plot_inputs, 1), use_dropout, test_time)

        # Plot data and functions.
        plt.cla()
        plt.title('Overfitting test with dropout rate = %f' % dropout_rate)
        ax.set_xlabel("Possible Inputs")
        ax.set_ylabel("Neural Network Outputs")
        ax.plot(inputs, targets, 'bx', label='Train Data')
        ax.plot(testinputs, testtargets, 'bo', label='Test Data')
        ax.plot(plot_inputs, outputs)
        ax.legend()
        ax.set_ylim([-2, 3])

        # # Plot the cost function for the test
        testcost = np.sum((neural_net_predict(params, testinputs, use_dropout, test_time) - testtargets)**2)
        traincost = np.sum((neural_net_predict(params, inputs, use_dropout, test_time) - targets)**2)
        #diff = np.absolute(testcost - traincost)
        testcostlist_dropout.append(testcost)
        traincostlist.append(traincost)
        iterlist.append(iter)
        ax2.plot(iterlist, testcostlist_dropout, 'r-', label='Test Cost')
        ax2.plot(iterlist, traincostlist, 'g-', label='Train Cost')
        ax2.set_xlabel('Number of Iterations')
        ax2.set_ylabel('Error in Estimation or Cost')
        ax2.set_xlim([0, 200])
        ax2.set_ylim([0, 20])
        if iter == 0:
            ax2.legend()
        
        plt.draw()
        #plt.savefig(str(dropout_rate*10) + str(iter) + '.jpg')
        plt.pause(1.0/60.0)

    def print_function(params, iter, gradient):
        use_dropout = False
        testcost = np.sum((neural_net_predict(params, testinputs, use_dropout) - testtargets)**2)
        testcostlist.append(testcost)

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=200, callback=print_function)
    optimized_params_dropout = adam(objective_grad_dropout, init_params, step_size=step_size,
                            num_iters=200, callback=print_function_dropout)


    # Set up figures.
    fig2 = plt.figure(figsize=(12, 8), facecolor='white')
    plt.title('Standard Test Cost vs. Dropout Test Cost, dropout rate=' + str(dropout_rate))
    ax3 = fig2.add_subplot(111)
    ax4 = fig2.add_subplot(111)
    ax3.plot(iterlist, testcostlist_dropout, '-r', label="Dropout Test Cost")
    ax4.plot(iterlist, testcostlist, '-b', label="Standard Test Cost")
    ax3.legend()
    ax4.legend()
    plt.ion()
    plt.show(block=False)

















