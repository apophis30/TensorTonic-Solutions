import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x)
    mask = x >= 0
    return x*mask