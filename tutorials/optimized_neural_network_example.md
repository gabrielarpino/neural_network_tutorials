# Optimized and Generalized Neural Network Example

Now, we will generalize our notion of a neural network. In our previous tutorial, "Simple Neural Network For Binary Classification Example", we looked at a simple method of developing a neural network for binary classification and training it using backpropagation.

Here, we will make two changes:

	*	Generalize the neural network to not just a binary classifier, but a system capable of learning any nonlinear function
	
	*	Formulate the network as an optimization problem, and solve it using an optimizer

Lets step through the code.

### Step 1: Define parameters, inputs, and targets

```python

    # Model parameters
    layer_sizes = [1,10,10,1]
    L2_reg = 0.01

    # Training parameters
    param_scale = 1.0
    step_size = 0.1
    inputs, targets = build_toy_dataset()

    # Randomly initialize the neural net parameters
    init_params = init_random_params(param_scale, layer_sizes)
```

Our inputs and targets are specified by the build_toy_dataset() function shown below:

```python

def build_toy_dataset(n_data=20, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets
```

It just creates two numpy arrays as sample inputs and targets following a function that takes as inputs numbers within (-1,-0.5),(0.5,1), and returns its cosine with some added noise. The inputs and targets would look as [such](../tutorials/sample_inputs_and_targets.md).

Next, the parameters are initialized through the init_random_params(param_scale, layer_sizes) function, which in short returns randomized initial parameters according to the chosen parameter scale and layer sizes, as [such](../tutorials/sample_init_params.md).

Confusing at first, but it is simply a list containing numpy arrays for the weights in each layer. Each layer contains two numpy arrays, one for the regular network neurons and another for the bias neuron.

### Step 2: Frame the network as an optimization problem

This section involves multiple function definitions. Hang tight because they all come together in the end.

```python

    # Define training objective
    def objective(params, iter):
        return -log_posterior(params, inputs, targets, L2_reg)
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

NOTE: We must negate the log posterior because we want to minimize the negative log of the probability, which is the same as maximizing the positive log of the probability. 

``` python

def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = -np.sum((neural_net_predict(params, inputs) - targets)**2)
    return log_prior + log_lik
```

The log_prior variable refers to the log(P(weights)) factor. This is estimated by taking the l2_norm of the parameters. The l2_norm is the famous euclidian magnitude, calculated as follows:

```python

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)
```

More on regularization in future tutorials.

The log_lik variable refers to the log(P(our outputs are correct | weights)) factor. This factor is determined by running a forward propagation on our network, subtracting the outputs from the targets, and squaring them. This gives us the squared magnitude of our error. The neural_net_predict function forward propagates as follows (surprisingly easy):

```python 

def neural_net_predict(params, inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return outputs
```

### Step 3: Find the derivative of the objective function

This is arguably the most important step to optimization: knowing which direction and magnitude with which to alter our weights. Thankfully, we depend on a package called autograd that amazingly takes derivatives of python code. It is able to work through loops, if statements, and function definitions involving numpy or scipy code. This next one line command is all we need to find the derivative of our objective function:

```python

    # Use autograd to obtain the gradient of the objective function
    objective_grad = grad(objective)
```

### Step 4: Optimize

We use an optimizing function called adam, and we minimize the objective function. Adam applies gradient descent with momentum. Short code for the optimization function can be viewed [here](../final_code/optimizers.py).

And that is all. We now have a neural network able to learn nonlinear functions, with no need for back propagation. The advantage to the use of autograd and the adam optimizer is that they keep the code concise and easily understandeable by framing the problem as an optimization problem.

Below are the results of running the complete [Code](../final_code/neural_net_optimized.py), the image below represents a graph of the neural network's performance in learning the approximate cosine function, where the target points are the X scatters on the plot:

<img src="neural_net_optimized.gif" width="400">

We can also view some of the network's weights in the 1st and 2nd hidden layer, to get a sense of the change that occurs within the network. Darker red means that that node has a larger outgoing weight:

<img src="network_weights.gif" width="400">