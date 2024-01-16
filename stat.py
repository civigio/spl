from math import sqrt, log


def mean(sample: list[float]) -> float:
    """
    Calculation of the mean of the sample present in the object

    Args:
        sample: list of floats representing data

    Returns:
        The mean of the sample
    """

    summ = sum(sample)
    n = len(sample)
    return summ / n


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def variance(sample: list[float],
             bessel: bool = True) -> float:
    """
    Calculation of the variance of the sample present in the object

    Args:
        sample: list of floats representing data
        bessel: applies the bessel correction (default true)

    Returns:
        The variance of the sample
    """

    summ = 0.
    sum_sq = 0.
    n = len(sample)
    for elem in sample:
        summ += elem
        sum_sq += elem * elem
    var = sum_sq / n - summ * summ / (n * n)
    if bessel:
        var = n * var / (n - 1)
    return var


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def stddev(sample: list[float],
           bessel: bool = True) -> float:
    """
    Calculation of the standard deviation of the sample present in the object

    Args:
        sample: list of floats representing data
        bessel: applies the bessel correction (default true)

    Returns:
        The standard deviation of the sample
    """

    return sqrt(variance(sample, bessel))


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def stderr(sample: list[float],
           bessel: bool = True) -> float:
    """
    Calculation of the standard error (standard deviation of the mean)
    of the sample present in the object

    Args:
        sample: list of floats representing data
        bessel: applies the bessel correction (default true)

    Returns:
        The standard error of the sample
    """
    n = len(sample)
    return sqrt(variance(sample, bessel) / n)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def likelihood(sample: list[float],
               parameter: float,
               pdf) -> float:
    """
    Calculation of the likelihood function for a sample
    of independent variables, identically distributed
    according to their pdf with a parameter

    Args:
        sample: list of floats representing data
        parameter: the parameter of the probability density function
        pdf: probability density function associated with the sample [must be expressed in the form pdf(x, parameter)]

    Returns:
        Value of the likelihood for the chosen parameter

    """

    result = 1.
    for x in sample:
        result = result * pdf(x, parameter)
    return result


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def loglikelihood(sample: list[float],
                  parameter: float,
                  pdf) -> float:
    """
    Calculation of the log-likelihood function for a sample
    of independent variables, identically distributed
    according to their pdf with a parameter

    Args:
        sample: list of floats representing data
        parameter: the parameter of the probability density function
        pdf: probability density function associated with the sample [must be expressed in the form pdf(x, parameter)]

    Returns:
        Value of the log-likelihood for the chosen parameter

    """

    result = 0.
    for x in sample:
        if pdf(x, parameter) > 0.:
            result = result + log(pdf(x, parameter))
    return result


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
