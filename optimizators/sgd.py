import numpy as np
import random
from losses.loss_functions import sum_of_squared_residuals


def gd(fn,x, y, w, b, learning_rate=0.1, iter_limit=500, tolerance=0.0001):
    cost1 = 10.0
    cost = 10.0
    for iteration in range(iter_limit):
        for x_, y_ in zip(x, y):
            # cost = (fn(x_, w, b) - y_) ** 2
              # this cost is called cross-entropy
            if y_ == 1:
                cost1 = -y_ * np.log(fn(x_, w, b)) - (1 - y_) * np.log(1 - fn(x_, w, b))
            else:
                cost = -y_ * np.log(fn(x_, w, b)) - (1 - y_) * np.log(1 - fn(x_, w, b))
            if cost <= tolerance and cost1 <= tolerance:
                print(fn(x_, w, b))
                return w, b
            w -= (1 / len(y)) * learning_rate * (fn(x_, w, b) - y_) * x_

            b -= (1 / len(y)) * learning_rate * (fn(x_, w, b) - y_)

    return w, b
