from typing import Callable
import numpy as np

def numerical_gradient(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, df: np.ndarray):
    """Computes the gradient of f at x using central differences.

    Args:
    - f (callable): A function that takes a NumPy array as input.
    - x (np.ndarray): The point at which to compute the gradient.
    - df (np.ndarray): The gradient of f at x.

    Returns:
    - gradient (np.ndarray): The gradient of f at x.
    """

    h = 1e-5
    gradient = np.zeros_like(x)

    iterator = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not iterator.finished:
        idx = iterator.multi_index
        original_value = x[idx]

        x[idx] = original_value + h
        fxph = f(x)

        x[idx] = original_value - h
        fxmh = f(x)

        gradient[idx] = np.sum((fxph - fxmh) * df) / (2 * h)
        x[idx] = original_value

        iterator.iternext()
    
    return gradient

def check_gradient(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, df: np.ndarray, grad_calculated) -> float:
    """Checks the implementation of a function that computes the gradient using central differences.

    Args:
    - f (callable): A function that takes a NumPy array as input.
    - x (np.ndarray): The point at which to check the gradient.
    - df (np.ndarray): The gradient of f at x.
    - grad_calculated (np.ndarray): The gradient of f at x, calculated using your implementation.

    Returns:
    - error (float): The error between df and grad_calculated.
    """
    
    numerical_grad = numerical_gradient(f, x, df)
    error = np.max(np.abs(numerical_grad - grad_calculated) / np.maximum(1e-8, np.abs(numerical_grad) + np.abs(grad_calculated)))
    return error