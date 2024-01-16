from generate import uniform_range, list_uniform_range
from math import sqrt


def hom(function,
        xmin: float,
        xmax: float,
        ymax: float,
        n_evt: int = 100000) -> tuple[float, float]:
    """
    Calculation of a defined integral of a function using the hit-or-miss method

    Args:
        function: function whose integral has to be calculated [must be expressed in the form function(x)]
        xmin: lower limit of the integral
        xmax: upper limit of the integral
        ymax: maximum value of the function in the interval
        n_evt: number of points generated to calculate the integral (optional, default: 100000)

    Returns:
        The defined integral and the uncertainty of the value obtained
    """

    x_coord = list_uniform_range(xmin, xmax, n_evt)
    y_coord = list_uniform_range(0., ymax, n_evt)

    points_under = 0
    for x, y in zip(x_coord, y_coord):
        if function(x) > y:
            points_under = points_under + 1

    area_rect = (xmax - xmin) * ymax
    frac = float(points_under) / float(n_evt)
    integral = area_rect * frac
    integral_unc = area_rect ** 2 * frac * (1 - frac) / n_evt
    return integral, sqrt(integral_unc)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def crude_mc(function,
             xmin: float,
             xmax: float,
             n_evt: int = 100000) -> tuple[float, float]:
    """
    Calculation of a defined integral of a function using the crude monte-carlo method

    Args:
        function: function whose integral has to be calculated [must be expressed in the form function(x)]
        xmin: lower limit of the integral
        xmax: upper limit of the integral
        n_evt: number of repetitions used to calculate the integral (optional, default: 100000)

    Returns:
        The defined integral and the uncertainty of the value obtained
    """

    summ = 0.
    squared_summ = 0.
    for i in range(n_evt):
        x = uniform_range(xmin, xmax)
        summ += function(x)
        squared_summ += function(x) * function(x)

    mean = summ / float(n_evt)
    variance = squared_summ / float(n_evt) - mean * mean
    variance = variance * (n_evt - 1) / n_evt
    length = (xmax - xmin)
    return mean * length, sqrt(variance / float(n_evt)) * length


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
