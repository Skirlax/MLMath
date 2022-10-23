from optimizators import sgd
from outputF import functions
import numpy as np
import random


def main():
    pass


if __name__ == "__main__":

    x = np.array([[5, 15, 25, 35, 45, 55], [15, 25, 35, 45, 55, 65]])
    y = np.array([0, 1, 14, 32, 22, 38])
    w = np.array([random.uniform(0,2) for _ in range(len(y))])
    b = 0.0
    w, b = sgd.gd(functions.logistic_regression, x, y, w, b)
    print(w, b)
    lg_result = functions.logistic_regression(x[1], w, b)
    formatted_ = '{:.2f}'.format(lg_result)
    print(formatted_)
