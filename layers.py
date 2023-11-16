from typing import Any, Dict, List, Tuple, Union
import numpy as np

def linear_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the forward pass for a linear layer.

    The input x has shape (N, D) which is a batch of size N.
    Each datapoint has D dimensions. We use a linear function to transform
    each datapoint to an output vector of dimension M.

    The output is to be calculated and 


    Args:
    - x: A numpy array containing input data, of shape (N, D)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns:
    - out: Output, of shape (N, M)
    - cache: Values requried for the backward pass.
    """

    out = np.dot(x,w) + b
    cache = (x, w, b)

    # TODO: Implement the linear forward pass and store the result in out.

    return out, cache

def linear_backward(dout: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the backward pass for a linear layer.

    Calculate the gradients using the formula derived with chain rule

    Args:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Values from the forward pass

    Returns:
    - dx: Gradient with respect to x, of shape (N, D)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """

    dx, dw, db = None, None, None
    x, w, b = cache
    m = x.shape[1]
    # TODO: Implement the linear backward pass and store the result in dx, dw, db.
    dx = np.dot(dout, w.T)     # Shape (N, D)
    dw = np.dot(x.T, dout)     # Shape (D, M)
    db = np.sum(dout, axis=0)  # Shape (M,)

    return dx, dw, db


def relu_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computs the output of a ReLU activation function.

    Args:
    - x (np.ndarray): A numpy array containing input data, of shape (N, D)

    Returns:
    - out (np.ndarray): Output, of shape (N, D)
    - cache (np.ndarray): Values requried for the backward pass.
    """

    out, cache = np.maximum(0, x), x

    # TODO: Implement the ReLU forward pass and store the result in out.

    return out, cache

def relu_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """Computes the backward pass for a ReLU activation function.

    Args:
    - dout (np.ndarray): Upstream derivative, of shape (N, D)
    - cache (np.ndarray): Values from the forward pass

    Returns:
    - dx (np.ndarray): Gradient with respect to x, of shape (N, D)
    """

    dx, x = None, cache

    # TODO: Implement the ReLU backward pass and store the result in dx.
    mask = (x > 0)
    # Multiply dout with the mask
    dx = dout * mask


    return dx



def cross_entropy_loss(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Computes the loss and gradient for softmax classification.

    Args:
    - logits (np.ndarray): Output prediction BEFORE softmax, of shape (N, C)
    - y (np.ndarray): A numpy array of labels, of shape (N,) where y[i] is the label for x[i] and 0 <= y[i] < C

    Returns:
    - loss (float): The loss value
    - dx (np.ndarray): Gradient with respect to x, of shape (N, C)
    """

    loss, dx = None, None

    # TODO: Implement the cross_entropy_loss forward pass and store the result in loss.
    # TODO: Implement the cross_entropy_loss backward pass and store the result in dx.
    N = logits.shape[0]

    # Apply softmax to logits
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute the loss
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N

    # Gradient calculation
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx


class NeuralNetwork:
    """Class for a Neural Network model with an arbitrary number of layers.

    The neural network will consist of
    - linear layers
    - activation functions (ReLU)
    - loss functions


    The parameters of the model will be stored in a dictionary (params) with the keys
    `W_0`, `b_0`, `W_1`, `b_1`, ..., `W_L-1`, `b_L-1` where `L` is the number of layers.

    There will be a ReLU activation function after each linear layer except for the last layer.



    This class will implement functions to perform the forward pass, the backward pass and the gradient descent update.
    """

    def __init__(self, hidden_dims: List[int], input_dims: int=28*28, num_classes: int=10):
        self.params: Dict[str, np.ndarray] = {} # Dictionary to hold the parameters of the model
        
        dims = [input_dims] + hidden_dims + [num_classes]

        for idx, (prev_dim, next_dim) in enumerate(zip(dims[:-1], dims[1:])):
            w, b = self.initialize_weights(prev_dim, next_dim)
            self.params[f'W_{idx}'] = w
            self.params[f'b_{idx}'] = b

    def initialize_weights(self, prev_dim: int, next_dim: int)  -> Tuple[np.ndarray, np.ndarray]:
        """Initializes the weights of a linear layer.

        The weights are initialized using Kaiming initialization.

        Args:
        - prev_dim (int): Dimensions of the previous layer
        - next_dim (int): Dimensions of the next layer

        Returns:
        - w (np.ndarray): The initialized weights
        - b (np.ndarray): The initialized biases
        """

        w = np.random.randn(prev_dim, next_dim) * np.sqrt(2 / prev_dim)
        b = np.zeros(next_dim)

        return w, b
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]]:
        """Performs the forward pass for the whole neural network.

        While implementing the forward pass, store the cache for each layer in the cache dictionary.
        The keys should be `linear_0`, `relu_0`, `linear_1`, ..., `linear_L-2`, `relu_L-2`, `linear_L-1`.

        The cache will be used to compute the gradients during the backward pass.

        Args:
        - x (np.ndarray): Input data of shape (N, D)

        Returns:
        - out (np.ndarray): Output of the last layer, of shape (N, C)
        - cache (Dict[str, ...]): Dictionary of caches for each layer
        """

        cache: Dict[str, np.ndarray] = {}
        out = x

        # TODO: Implement the forward pass for the whole neural network.
        for idx in range(len(self.params) // 2):
            w, b = self.params[f'W_{idx}'], self.params[f'b_{idx}']
            out, cache[f'linear_{idx}'] = linear_forward(out, w, b)
            out, cache[f'relu_{idx}'] = relu_forward(out)
        
        return out, cache
    
    def backward(self, dout: np.ndarray, cache: Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]) -> Dict[str, np.ndarray]:
        """Performs the backward pass for the whole neural network.

        The backward pass uses the cached values stored during the forward pass to compute the gradients.
        The gradients are stored in a dictionary with the same keys as the params dictionary.

        Args:
        - dout (np.ndarray): Upstream derivatives of shape (N, C)
        - cache (Dict[str, ...]): Dictionary of caches for each layer

        Returns:
        - grads (Dict[str, np.ndarray]): Dictionary of gradients for each layer
        """

        grads: Dict[str, np.ndarray] = {}

        # TODO: Implement the backward pass for the whole neural network and store the gradients in grads.
        num_layers = len(self.params) // 2

        # Backpropagation for the last layer
        dout, grads[f'W_{num_layers - 1}'], grads[f'b_{num_layers - 1}'] = linear_backward(dout, cache[f'linear_{num_layers - 1}'])

        # Backpropagation for the remaining layers
        for idx in range(num_layers - 2, -1, -1):
            dout = relu_backward(dout, cache[f'relu_{idx}'])
            dout, grads[f'W_{idx}'], grads[f'b_{idx}'] = linear_backward(dout, cache[f'linear_{idx}'])
        return grads
    
    def update(self, grads: Dict[str, np.ndarray], lr: float):
        """Performs gradient descent update on all the parameters.

        Args:
        - grads (Dict[str, np.ndarray]): Dictionary of gradients for each layer
        - lr (float): Learning rate
        """

        # TODO: Implement the gradient descent update for each parameter.
        for key in self.params.keys():
            self.params[key] -= lr * grads[key]
        pass

