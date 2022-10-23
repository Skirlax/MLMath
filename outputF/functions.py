import numpy as np
import math


def simple_linear_regression(x, w, b):
    return np.dot(x, w) + b


def logistic_regression(x,w,b):
    z = np.dot(x,w) + b
    return 1 / (1 + math.e ** (-z))
