import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x, dtype=np.float32)
    return np.where(x>=0, 1/(1+np.exp(-x)), np.exp(-np.abs(x))/(1+np.exp(-np.abs(x))))