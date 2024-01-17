
def bisection(function,
              x_min: float,
              x_max: float,
              precision: float = 0.0001) -> float:
    """
    Bisection method for finding zeros of a function. The variables x_min and x_max must have opposite sign

    Args:
        function: function to be studied [must be expressed in the form function(x)]
        x_min: lower limit of the interval of search
        x_max: upper limit of the interval of search
        precision: precision with which the zero is found (optional, default: 0.0001)

    Returns:
        The zero value of a function up to the decimal established in precision
    """

    x_ave = x_min
    while (x_max - x_min) > precision:
        x_ave = 0.5 * (x_max + x_min)
        if function(x_ave) * function(x_min) > 0.:
            x_min = x_ave
        else:
            x_max = x_ave
    return x_ave


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def golden_ratio(function,
                 x_min: float,
                 x_max: float,
                 precision: float = 0.0001,
                 minimum: bool = True) -> float:
    """
    Golden ratio method for finding minimum or maximum of a function in the
    interval [x_min, x_max] with a certain precision

    Args:
        function: function to be studied [must be expressed in the form function(x)]
        x_min: lower limit of the interval of search
        x_max: upper limit of the interval of search
        precision: precision with which the maximum or minimum is found (optional, default: 0.0001)
        minimum: if True, the function calculates the minimum; if False, the function calculates the maximum
            (optional, default: True)

    Returns:
        The minimum or maximum of the function
    """

    ratio = 0.618
    x1 = x_max - (x_max - x_min) * ratio
    x2 = x_min + (x_max - x_min) * ratio

    f1 = function(x1)
    f2 = function(x2)

    while abs(x_max - x_min) > precision:
        if (minimum and f1 < f2) or (not minimum and f1 > f2):
            x_max = x2
            x2 = x1
            x1 = x_max - (x_max - x_min) * ratio
            f2 = f1
            f1 = function(x1)
        else:
            x_min = x1
            x1 = x2
            x2 = x_min + (x_max - x_min) * ratio
            f1 = f2
            f2 = function(x2)

    return (x_min + x_max) / 2


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
