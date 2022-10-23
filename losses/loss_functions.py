import numpy as np


def sum_of_squared_residuals(x,y,w,b,fn):
    cost_sum = 0.0
    for x_,y_ in zip(x,y):
        cost_sum += (fn(x,w,b) - y_) ** 2
    return cost_sum


def cross_entropy(x,y,w,b,fn):
    cost_sum = 0.0
    for x_,y_ in zip(x,y):
        cost_sum += -y_ * np.log(fn(x_,w,b)) - (1 - y_) * np.log(1 - fn(x_,w,b))
    return cost_sum

