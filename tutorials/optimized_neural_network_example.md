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

It just creates two numpy arrays as sample inputs and targets following a function that takes as inputs numbers within (-1,-0.5),(0.5,1), and returns its cosine with some added noise. The inputs and targets would look as such:

```
	Inputs 

	array([[-1.        ],				
       [-0.94444444],
       [-0.88888889],
       [-0.83333333],
       [-0.77777778],
       [-0.72222222],
       [-0.66666667],
       [-0.61111111],
       [-0.55555556],
       [-0.5       ],
       [ 0.5       ],
       [ 0.55555556],
       [ 0.61111111],
       [ 0.66666667],
       [ 0.72222222],
       [ 0.77777778],
       [ 0.83333333],
       [ 0.88888889],
       [ 0.94444444],
       [ 1.        ]])


   --

   	Targets

   	array([[  1.17640523e+00],
       [  1.01542581e+00],
       [  1.00072347e+00],
       [  1.00997658e+00],
       [  8.17030850e-01],
       [  3.45938234e-01],
       [  3.30246415e-01],
       [  1.04460393e-04],
       [ -2.15828606e-01],
       [ -3.75086986e-01],
       [  9.74574644e-01],
       [  1.14356968e+00],
       [  1.06312967e+00],
       [  9.39535205e-01],
       [  8.66488047e-01],
       [  7.09772354e-01],
       [  6.46850548e-01],
       [  2.73500390e-01],
       [  1.07436895e-01],
       [ -2.30909608e-01]])
```

Next, the parameters are initialized through the init_random_params(param_scale, layer_sizes) function, which in short returns randomized initial parameters according to the chosen parameter scale and layer sizes, as such:

```python
>>> init_params
[(array([[ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,
        -0.97727788,  0.95008842, -0.15135721, -0.10321885,  0.4105985 ]]), array([ 0.14404357,  1.45427351,  0.76103773,  0.12167502,  0.44386323,
        0.33367433,  1.49407907, -0.20515826,  0.3130677 , -0.85409574])), (array([[-2.55298982,  0.6536186 ,  0.8644362 , -0.74216502,  2.26975462,
        -1.45436567,  0.04575852, -0.18718385,  1.53277921,  1.46935877],
       [ 0.15494743,  0.37816252, -0.88778575, -1.98079647, -0.34791215,
         0.15634897,  1.23029068,  1.20237985, -0.38732682, -0.30230275],
       [-1.04855297, -1.42001794, -1.70627019,  1.9507754 , -0.50965218,
        -0.4380743 , -1.25279536,  0.77749036, -1.61389785, -0.21274028],
       [-0.89546656,  0.3869025 , -0.51080514, -1.18063218, -0.02818223,
         0.42833187,  0.06651722,  0.3024719 , -0.63432209, -0.36274117],
       [-0.67246045, -0.35955316, -0.81314628, -1.7262826 ,  0.17742614,
        -0.40178094, -1.63019835,  0.46278226, -0.90729836,  0.0519454 ],
       [ 0.72909056,  0.12898291,  1.13940068, -1.23482582,  0.40234164,
        -0.68481009, -0.87079715, -0.57884966, -0.31155253,  0.05616534],
       [-1.16514984,  0.90082649,  0.46566244, -1.53624369,  1.48825219,
         1.89588918,  1.17877957, -0.17992484, -1.07075262,  1.05445173],
       [-0.40317695,  1.22244507,  0.20827498,  0.97663904,  0.3563664 ,
         0.70657317,  0.01050002,  1.78587049,  0.12691209,  0.40198936],
       [ 1.8831507 , -1.34775906, -1.270485  ,  0.96939671, -1.17312341,
         1.94362119, -0.41361898, -0.74745481,  1.92294203,  1.48051479],
       [ 1.86755896,  0.90604466, -0.86122569,  1.91006495, -0.26800337,
         0.8024564 ,  0.94725197, -0.15501009,  0.61407937,  0.92220667]]), array([ 0.37642553, -1.09940079,  0.29823817,  1.3263859 , -0.69456786,
       -0.14963454, -0.43515355,  1.84926373,  0.67229476,  0.40746184])), (array([[-0.76991607],
       [ 0.53924919],
       [-0.67433266],
       [ 0.03183056],
       [-0.63584608],
       [ 0.67643329],
       [ 0.57659082],
       [-0.20829876],
       [ 0.39600671],
       [-1.09306151]]), array([-1.49125759]))]
```

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

The next issue we are faced with is convexity, or whether the optimization problem has one single minimum. In order to convexify this function (rid it of its possibly misleading local minima), we apply log() to both sides of the equation. Intuitively, the log function increases by a larger amount on differential changes in small variables, and increases by a smaller amount on differential changes in already large variables. The derivative of log(x) is d/dx (log(x)) = 1/x, 1/x being very large and positive when x is close to 0 (greater derivative when x is small), and small and positive otherwise. Applying log(), the equation becomes:

	log(P (our weights are correct | outputs)) ~ log(P (our outputs are correct | weights)) + log(P(weights))

And this is exactly what the log_posterior function is referring to!, except that we must negate the log posterior and all of its as we are wanting to maximize the probability of our weights being correct. Lets take a look into it:

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