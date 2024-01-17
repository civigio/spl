
def bisection(function,
              x_min: float,
              x_max: float,
              precision: float = 0.0001) -> float:
    """
    Bisection method for finding zeros of a function. The variables x_min and x_max must have opposite sign

    Args:
        function: function to be studied
        x_min: lower limit of the interval of search
        x_max: upper limit of the interval of search
        precision: precision with which the zero is found

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
