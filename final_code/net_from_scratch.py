import autograd.numpy as np
from autograd import elementwise_grad
from autograd import grad
import matplotlib
import matplotlib.pyplot as plt

#Define the objective function (sigmoid)
def objective(a):
	return 1/(1 + np.exp(a))

#Define the input arrays x and the desired output array y
x = np.array([[1,1,0]
		,[1,0,0]
		,[1,1,1]
		,[0,1,1]])
y = np.array([[0,0,1,1]]).T

#Declare the desired learning rate for the network
learning_rate = 1

np.random.seed(1)
#create weight vectors with average 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,4)) - 1
syn2 = 2*np.random.random((4,1)) - 1

# Set up figure.
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, frameon=False)
ax.set_ylim([-0.5, 0.5])
ax.set_xlim([0, 4])
plt.ion()
plt.show()
count = 0

for iter in xrange(180):

	#Forward Propagation
	l0 = x
	l1 = objective(np.dot(l0,syn0))
	l2 = objective(np.dot(l1,syn1))
	l3 = objective(np.dot(l2,syn2))

	objective_deriv = elementwise_grad(objective)

	#Backward Propagation
	delta_output = y - l3

	delta_l3 = delta_output*objective_deriv(l3)
	l2_error = np.dot(delta_l3, syn2.T)	

	delta_l2 = l2_error*objective_deriv(l2)
	l1_error = np.dot(delta_l2, syn1.T)

	delta_l1 = l1_error*objective_deriv(l1)

	#Adjust the neural synapses according to the errors produced
	syn2 += learning_rate*l2.T.dot(delta_l3)
	syn1 += learning_rate*l1.T.dot(delta_l2)
	syn0 += learning_rate*l0.T.dot(delta_l1)

	#Plot the error as it changes during the back-propagation steps
	count = count + 1
	plt.cla()
	ax.plot(delta_output)
	plt.ylabel('Error')
	plt.xlabel('Output Vector Index')
	plt.draw()

	plt.pause(1.0/100000000.0)

