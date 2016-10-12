from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad

def nonlin(a):
	return 1/(1 + np.exp(a))

x = np.linspace(-7,7,200)
derivative = elementwise_grad(nonlin)
plt.plot(x,nonlin(x),x,derivative(x))
plt.show()