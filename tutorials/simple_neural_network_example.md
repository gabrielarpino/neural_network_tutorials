#Simple Neural Network Example

Let's implement a simple neural network from scratch with 4 layers total: one for the input, two that are hidden, and one for the output. The inputs to our neural network will be the following:

	Inputs 						|Outputs
	[1,1,0]						 0
	[1,0,0]						 0
	[1,1,1]						 1
	[0,1,1]						 1

It is easy to note that there is a direct correlation between the 3rd index of the input vector and the output: when the 3rd value of the vector is 1, the output is 1, and when the 3rd value of the vector is 0, the output is 0! Let's see if a neural network can figure this out. We will use a back propagation algorithm in order to train the inputs into outputs. 

### Step 1: Define an Objective function, or the sigmoid function to be used at every neuron:

```python

#Define the objective function (sigmoid)
def objective(a):
	return 1/(1 + np.exp(a))

```

Let's proceed by defining our initial inputs and weight vectors.

### Step 2: Input/Output definition and initialization

```python

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
```

All that is left to do is to run forward and backwards propagation. We do this using a for loop.

### Step 3: Forwards and Backwards propagation

```python

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

```

And we are done! After this code has ran, the synapses have been adjusted and the parameters are thefore tuned to reproduce the ouputs specified by the targets. Joining the parts of the code and adding some plotting features, the final code can be viewed [here.](net_from_scratch.py)

Error for this neural network was plotted for every input, showing that the error (the difference between the neural network's prediction and the target values) goes to zero for all four inputs. The plot is displayed with Error on the Y-axis and Output Vector Index on the X-axis:

<img src="neural_net_backprop.gif" width="400">
