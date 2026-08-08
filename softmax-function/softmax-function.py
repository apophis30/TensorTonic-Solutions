import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asarray(x)
    x_max = x.max(axis=-1, keepdims=True)
    return np.exp(x-x_max)/np.sum(np.exp(x-x_max), axis=-1, keepdims=True)