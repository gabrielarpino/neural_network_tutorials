# Neural Network in 20 lines of python

[Full code here](../final_code/concise_optimized_neural_net.py).

Here, we will analyze a neural network with the following properties:

	*	The neural network will be generalized, not only a binary classifier but a system capable of learning any nonlinear function
	
	*	The network is framed as an optimization problem, allowing for efficient training.

Lets step through the code.

### Step 1: Define our network prediction function.

This is where we forward propagate our inputs through the multilayer perceptron. As seen below, this is done through one simple three line for loop!

```python 

def neural_net_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs
```

The loop multiplies the incoming perceptron values by the appropriate weights, adds the bias term b, and passes that through a sigmoid function tanh(x). It then defines inputs as the next layer of perceptrons, and repeats the steps for all layers.

Of course, the params and inputs matrices have to be formatted correctly for this concise forward propagation to work. Let's do that:

### Step 2: Define parameters, inputs, and targets

```python

    # Model and training parameters
    layer_sizes = [1,10,10,1]
    param_scale, step_size = 1.0, 0.1
    inputs = np.array([[-1.0],[-0.875],[-0.75],[-0.625],[-0.5],[0.5],[0.625],[0.75],[0.875],[1.0]])
    targets = np.array([[ 1.17],[ 0.92],[ 0.64],[ 0.30],[-0.23],[0.86],[1.07],[0.74],[0.34],[-0.10]])

    # Randomly initialize the neural net parameters
    init_params = init_random_params(param_scale, layer_sizes)
```

These inputs and targets are random points from the range -1 to 1 of the cos(x) function.

Next, the parameters are initialized through the init_random_params(param_scale, layer_sizes) function, which in short returns randomized initial parameters according to the chosen parameter scale and layer sizes, as [such](../tutorials/sample_init_params.md).

Confusing at first, but it is simply a list containing numpy arrays for the weights in each layer. Each layer contains two numpy arrays, one for the regular network neurons and another for the bias neuron.

### Step 3: Frame the network as an optimization problem

This section involves multiple function definitions. Hang tight because they all come together in the end.

```python

    # Define training objective, equivalent to the log_posterior of the distribution
    def objective(params, iter):
        return np.sum((neural_net_predict(params, inputs) - targets)**2)
```

We begin by specifying our objective function, or function to be optimized. As we know from bayesian probability, the function we are aiming to optimize is the following:

	P (our weights are correct | outputs) = P (our outputs are correct | weights) * P(weights)/P(outputs)

	(From baye's rule, P (x|y) = P (y|x) * P(x)/P(y)) 

Now, given that we want to maximize the probability of our weights being correct, we can rid ourselves of the P(outputs) factor in the denominator and only deal with the numerator. The P(outputs) factor is actually just a normalizing factor, and it is the constant theoretical probability of producing an output (too abstract to accurately deduce). Now ridding ourselves of this factor simplifies the relation:

	P (our weights are correct | outputs) ~ P (our outputs are correct | weights) * P(weights)

Where ~ denotes proportionality. 

The next issue we are faced with is convexity, or whether the optimization problem has one single minimum. In order to convexify this function (rid it of its possibly misleading local minima), we apply log() to both sides of the equation. Intuitively, the log function increases by a larger amount on differential changes in small variables, and increases by a smaller amount on differential changes in already large variables. The derivative of log(x) is d/dx (log(x)) = 1/x, 1/x being very large and positive when x is close to 0 (greater derivative when x is small), and small otherwise. Applying log(), the equation becomes:

	log(P (our weights are correct | outputs)) ~ log(P (our outputs are correct | weights)) + log(P(weights))

And this is exactly what the log_posterior function is referring to!

The log(P(weights)) is used for a concept called regularization, explored [here](../tutorials/regularization_example.md). We omit it from this network for simplicity.

### Step 4: Find the derivative of the objective function

This is arguably the most important step to optimization: knowing which direction and magnitude with which to alter our weights. Thankfully, we depend on a package called autograd that amazingly takes derivatives of python code. It is able to work through loops, if statements, and function definitions involving numpy or scipy code. This next one line command is all we need to find the derivative of our objective function:

```python

    # Use autograd to obtain the gradient of the objective function
    objective_grad = grad(objective)
```

### Step 5: Optimize

We use an optimizing function called adam, and we minimize the objective function. Adam applies gradient descent with momentum. Short code for the optimization function can be viewed [here](../final_code/optimizers.py).

And that is all. We now have a neural network able to learn nonlinear functions, with no need for back propagation. The advantage to the use of autograd and the adam optimizer is that they keep the code concise and easily understandeable by framing the problem as an optimization problem.

The [concise code can be found here](../final_code/concise_optimized_neural_net.py), which outputs the total error of the network in fitting the function.

Below are the results of running the [visual and fully commented version of the code](../final_code/neural_net_optimized.py), the image below represents a graph of the neural network's performance in learning the approximate cosine function, where the target points are the X scatters on the plot:

<img src="neural_net_optimized.gif" width="400">

A similar writeup for the visual and fully commented version of the code can be found [here](../tutorials/optimized_neural_network_example.md).

We can also view some of the network's weights in the 1st and 2nd hidden layer, to get a sense of the change that occurs within the network. Darker red means that that node has a larger outgoing weight:

<img src="network_weights.gif" width="400">