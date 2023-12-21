import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import time
import random
from iminuit import Minuit
from iminuit.cost import LeastSquares
from IPython.display import display


def list_uniform(n: int,
                 seed: float = 0.) -> list[float]:
    """
    Generation of a list of n pseudo-casual number distributed accordingly
    to uniform distribution between [0, 1) starting from an optional seed
    different from 0.

    Args:
        n: number of pseudo-casual numbers to generate
        seed: starting seed for the random generation (optional)

    Returns:
        A list of n pseudo-casual numbers generated according to uniform distribution between [0, 1)
    """

    if seed != 0.:
        random.seed(seed)
    random_list = []
    for i in range(n):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        random_list.append(random.random())
    return random_list


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def uniform_range(minimum: float,
                  maximum: float) -> float:
    """
    Generation of a pseudo-casual number distributed accordingly to uniform distribution between
    [minimum, maximum)

    Args:
        minimum: lower limit of the range (included)
        maximum: upper limit of the range (excluded)

    Returns:
        A pseudo-casual numbers generated according to uniform distribution between [minimum, maximum)
    """

    return minimum + (random.random() * (maximum - minimum))


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def list_uniform_range(minimum: float,
                       maximum: float,
                       n: int,
                       seed: float = 0.) -> list[float]:
    """
        Generation of a list of n pseudo-casual number distributed accordingly
        to uniform distribution between [minimum, maximum) starting from an optional seed
        different from 0.

        Args:
            minimum: lower limit of the range (included)
            maximum: upper limit of the range (excluded)
            n: number of pseudo-casual numbers to generate
            seed: starting seed for the random generation (optional)

        Returns:
            A list of n pseudo-casual numbers generated according to uniform distribution between [minimum, maximum)
        """

    if seed != 0.:
        random.seed(seed)
    random_list = []
    for i in range(n):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        random_list.append(uniform_range(minimum, maximum))
    return random_list


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def tcl_ms(mean: float,
           sigma: float,
           n_sum: int = 10) -> float:
    """
    Generation of a pseudo-casual number distributed accordingly to the gaussian distribution
    with the central limit theorem algorithm between known mean value and standard deviation

    Args:
        mean: mean value
        sigma: standard deviation
        n: number of repetitions used in the algorithm (optional, default: 10)

    Returns:
        A pseudo-casual numbers generated according to gaussian distribution specified

    """

    y = 0.
    delta = math.sqrt(3 * n_sum) * sigma
    minimum = mean - delta
    maximum = mean + delta
    for i in range(n_sum):
        y += uniform_range(minimum, maximum)
    y /= n_sum
    return y


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def list_tcl_ms(mean: float,
                sigma: float,
                n: int,
                n_sum: int = 10,
                seed: float = 0.) -> list[float]:
    """
    Generation of a list of n pseudo-casual numbers distributed accordingly to the gaussian distribution
    with the central limit theorem algorithm between known mean value and standard deviation starting from an optional seed
    different from 0.

    Args:
        mean: mean value
        sigma: standard deviation
        n: number of repetitions used in the algorithm (optional, default: 10)

    Returns:
        A pseudo-casual numbers generated according to gaussian distribution specified

    """

    y = 0.
    delta = math.sqrt(3 * n) * sigma
    minimum = mean - delta
    maximum = mean + delta
    for i in range(n):
        y += uniform_range(minimum, maximum)
    y /= n
    return y


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
