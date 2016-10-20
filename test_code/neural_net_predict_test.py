from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import flatten
from optimizers import adam

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
    	print('W')
    	print(W)
    	print('b')
    	print(b)
    	print('inputs')
        print(inputs)
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
        print('outputs')
        print(outputs)
        print('inputs')
        print(inputs)

        print('logsumexp(outputs, keepdims=True)')
        print(logsumexp(outputs, keepdims=True))
        print('done')
    return outputs - logsumexp(outputs, keepdims=True)


if __name__ == '__main__':

	# Build a toy dataset.
	inputs = np.array([[0.52, 1.12,  0.77],
						[0.34, 1.22, 0.22]])
	targets = np.array([True, True, False, True])
	weights = np.array([([1.0, 2.0 , 3.0],0.0), 
						([0.99, 0.88, 0.77], 1.0)])

	print(neural_net_predict(weights, inputs))