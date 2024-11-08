from numpy.random import exponential


def exp_delay(x, y=None):
    return exponential(scale=x, size=y)
