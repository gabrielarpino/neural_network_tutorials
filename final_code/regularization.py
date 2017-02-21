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
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = -np.sum((neural_net_predict(params, inputs) - targets)**2)
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
    L2_reg = 0.3

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

    # Use autograd to obtain the gradient of the objective function
    objective_grad = grad(objective)

    # Set up figures.
    fig1 = plt.figure(figsize=(12, 8), facecolor='white')
    ax2 = fig1.add_subplot(212, frameon=False)
    ax = fig1.add_subplot(211, frameon=False)
    plt.ion()
    plt.show(block=False)

    testcostlist = []
    traincostlist = []
    iterlist = []

    def print_function(params, iter, gradient):
            plot_inputs = np.linspace(-8, 8, num=400)
            outputs = neural_net_predict(params, np.expand_dims(plot_inputs, 1))

            # Plot data and functions.
            plt.cla()
            plt.title('Overfitting test with L2_reg = %f' % L2_reg)
            ax.set_xlabel("Possible Inputs")
            ax.set_ylabel("Neural Network Outputs")
            ax.plot(inputs, targets, 'bx', label='Train Data')
            ax.plot(testinputs, testtargets, 'bo', label='Test Data')
            ax.plot(plot_inputs, outputs)
            ax.legend()
            ax.set_ylim([-2, 3])

            # Plot the cost function for the test
            testcost = np.sum((neural_net_predict(params, testinputs) - testtargets)**2)
            traincost = np.sum((neural_net_predict(params, inputs) - targets)**2)
            diff = np.absolute(testcost - traincost)
            testcostlist.append(testcost)
            traincostlist.append(traincost)
            iterlist.append(iter)
            ax2.plot(iterlist, testcostlist, 'r-', label='Test Cost')
            ax2.plot(iterlist, traincostlist, 'g-', label='Train Cost')
            ax2.set_xlabel('Number of Iterations')
            ax2.set_ylabel('Error in Estimation or Cost')
            ax2.set_xlim([0, 50])
            if iter == 0:
                ax2.legend()
            
            plt.draw()
            #plt.savefig(str(iter) + '.jpg')
            plt.pause(1.0/60.0)

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=300, callback=print_function)



