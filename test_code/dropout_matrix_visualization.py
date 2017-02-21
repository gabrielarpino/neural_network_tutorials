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

def neural_net_predict(params, inputs, dropout = True):
    """Implements a deep neural network for classification. params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix. returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
        if dropout: np.random.binomial([np.ones_like(inputs)],0.8)[0]/(0.8)
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

    # Set up figure.
    fig1 = plt.figure(figsize=(12, 8), facecolor='white')
    ax = plt.subplot2grid((4,3),(0, 0), colspan = 3)
    ax2 = plt.subplot2grid((4,3),(3, 0))
    ax3 = plt.subplot2grid((4,3),(2, 0))
    ax4 = plt.subplot2grid((4,3),(3, 1))
    ax5 = plt.subplot2grid((4,3),(2, 1))
    ax6 = plt.subplot2grid((4,3),(3, 2))
    ax7 = plt.subplot2grid((4,3),(2, 2))
    plt.ion()
    plt.show(block=False)

    def print_function(params, iter, gradient):
            """ Plot data and functions. """
            plot_inputs = np.linspace(-8, 8, num=400)
            outputs = neural_net_predict(params, np.expand_dims(plot_inputs, 1), False)
            ax.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
            ax5.cla()
            ax6.cla()
            ax7.cla()
            ax.plot(inputs, targets, 'bx')
            ax.plot(plot_inputs, outputs)
            ax.set_xlabel("Possible Inputs")
            ax.set_ylabel("Neural Network Outputs")
            ax.set_ylim([-2,2])
            plt.draw()
            ax2.matshow(params[0][0].T, cmap='cool')
            ax2.set_xlabel("Hidder Layer 1 (Incoming Weights)")
            ax3.matshow(np.array([params[0][1]]).T, cmap='cool')
            ax3.set_ylabel("Hidder Layer 1 Bias")
            ax4.matshow(params[1][0].T, cmap='cool')
            ax4.set_xlabel("Hidden Layer 2 (Incoming Weights)")
            ax5.matshow(np.array([params[1][1]]).T, cmap='cool')
            ax5.set_ylabel("Hidder Layer 2 Bias")
            ax6.matshow(params[2][0], cmap='cool')
            ax6.set_xlabel("Hidden Layer 2 (Outgoing weights)")
            ax7.matshow(np.array([params[2][1]]), cmap='cool')
            ax7.set_ylabel("Output Bias")
            #plt.savefig(str(iter) + '.jpg')
            plt.pause(1.0/60.0)

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad, init_params, step_size=step_size,
                            num_iters=100, callback=print_function)

