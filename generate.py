import numpy as np
import math
import random


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


def clt_ms(mean: float,
           sigma: float,
           n_sum: int = 10) -> float:
    """
    Generation of a pseudo-casual number distributed accordingly to the gaussian distribution
    with the central limit theorem algorithm between known mean value and standard deviation

    Args:
        mean: mean value
        sigma: standard deviation
        n_sum: number of repetitions used in the algorithm (optional, default: 10)

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


def list_clt_ms(mean: float,
                sigma: float,
                n: int,
                n_sum: int = 10,
                seed: float = 0.) -> list[float]:
    """
    Generation of a list of n pseudo-casual numbers distributed accordingly to the gaussian distribution
    with the central limit theorem algorithm known mean value and standard deviation starting from an optional seed
    different from 0.

    Args:
        mean: mean value
        sigma: standard deviation
        n: length of the list
        n_sum: number of repetitions used in the algorithm (optional, default: 10)
        seed: starting seed for the random number generator (optional, default: 0.)

    Returns:
        A list of pseudo-casual numbers generated according to gaussian distribution specified

    """

    if seed != 0.:
        random.seed(seed)
    random_list = []
    for i in range(n):
        random_list.append(clt_ms(mean, sigma, n_sum))
    return random_list


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def clt_minmax(minimum: float,
               maximum: float,
               n_sum: int = 10) -> float:
    """
    Generation of a pseudo-casual number with the central limit theorem algorithm
    between [minimum, maximum)

    Args:
        minimum: lower limit of the range (included)
        maximum: upper limit of the range (excluded)
        n_sum: number of repetitions used in the algorithm (optional, default: 10)

    Returns:
        A pseudo-casual numbers generated with the central limit theorem algorithm
        between [minimum, maximum)
    """

    y = 0.
    for i in range(n_sum):
        y += uniform_range(minimum, maximum)
    y /= n_sum
    return y


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def list_clt_minmax(minimum: float,
                    maximum: float,
                    n: int,
                    n_sum: int = 10,
                    seed: float = 0.) -> list[float]:
    """
    Generation of a list of n pseudo-casual numbers distributed between [minimum, maximum)
    with the central limit theorem algorithm starting from an optional seed
    different from 0.

    Args:
        minimum: lower limit of the range (included)
        maximum: upper limit of the range (excluded)
        n: length of the list
        n_sum: number of repetitions used in the algorithm (optional, default: 10)
        seed: starting seed for the random number generator (optional, default: 0.)

    Returns:
        A list of pseudo-casual numbers generated according to gaussian distribution specified

    """

    if seed != 0.:
        random.seed(seed)
    random_list = []
    for i in range(n):
        random_list.append(clt_minmax(minimum, maximum, n_sum))
    return random_list


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def list_ifm_exponential(t_0: float,
                         n: int) -> list[float]:
    """
    Generation of a list of n pseudo-casual numbers distributed accordingly
    to an exponential distribution with a characteristic time t_0 with the inverse function method

    Args:
        t_0: characteristic time of the exponential distribution
        n: length of the list

    Returns:
        A list of pseudo-casual numbers generated according to an exponential distribution
        with a characteristic time t_0
    """

    random_list = []
    for i in range(n):
        random_list.append(-((np.log(1-random.random()))*t_0))
    return random_list


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def list_ifm_poisson(lambda_value: float,
                     n: int) -> list[float]:
    """
    Generation of a list of n pseudo-casual numbers distributed accordingly
    to a poissonian distribution with an expected value lambda_value with the inverse function method
    combined with a monte-carlo/toy experiments technique

    Args:
        lambda_value: expected value of the poissonian distribution
        n: length of the list

    Returns:
        A list of pseudo-casual numbers generated according to a poissonian distribution with
        an expected value lambda_value
    """

    random_list = []
    for j in range(n):
        i = 0
        delta = 0
        while delta <= lambda_value:
            n = -(np.log(1 - random.random()))
            delta = delta + n
            i = i + 1
        random_list.append(i - 1)

    return random_list


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def tac_box(function,
            x_minimum: float,
            x_maximum: float,
            y_minimum: float,
            y_maximum: float,
            seed: float = 0.) -> float:
    """
    Generation of a pseudo-casual number distributed accordingly to a function
    with the try-an-catch algorithm, into a "box" delimited by x_minimum and x_maximum for the
    horizontal axis and y_minimum and y_maximum for the vertical axis starting from an optional seed
    different from 0.

    Args:
        function: the function which rules the distribution of numbers
        x_minimum: lower limit of the range for the horizontal axis
        x_maximum: upper limit of the range for the horizontal axis
        y_minimum: lower limit of the range for the vertical axis
        y_maximum: upper limit of the range for the vertical axis
        seed: starting seed for the random generation (optional)

    Returns:
        A single number generated with the try-an-catch algorithm distributed accordingly to the function wanted
    """

    if seed != 0.:
        random.seed(seed)
    x = uniform_range(x_minimum, x_maximum)
    y = uniform_range(y_minimum, y_maximum)
    while y > function(x):
        x = uniform_range(x_minimum, x_maximum)
        y = uniform_range(y_minimum, y_maximum)
    return x


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
