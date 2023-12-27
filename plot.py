import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import time
import random
from iminuit import Minuit
from iminuit.cost import LeastSquares
from IPython.display import display


def histogram(sample: list[float],
              title: str = 'Histogram',
              xlabel: str = 'x-axis',
              ylabel: str = 'y-axis',
              label: str = 'Histogram'):
    """
    Plots a histogram of samples, with optional title and x-label and y-label and legend. The function
    saves the histogram as a png image

    Args:
        sample: list of floats representing data
        title: title of the histogram
        xlabel: label of the x-axis
        ylabel: label of the y-axis
        label: title of the histogram in the legend

    Returns:
        The plot of the histogram of the sample
    """

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(sample, label=label)
    ax.set_title(title, size=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    plt.savefig(title + '.png')
    plt.show()
    return


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def scatter():
    return


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----


def graph():
    return


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
