import numpy as np
from iminuit import Minuit
from math import floor, ceil
from iminuit.cost import UnbinnedNLL
from matplotlib import pyplot as plt
from scipy.stats import norm
from IPython.display import display


def pdf(x, mu, sigma):
    return norm.pdf(x, mu, sigma)


def main():
    data = np.loadtxt("data/dati_2.txt")

    bin_content, bin_edges = np.histogram(data, bins=(floor(len(data) / 100)), range=(floor(min(data)), ceil(max(data))))

    sample_mean = np.mean(data)
    sample_sigma = np.std(data)

    fig, ax = plt.subplots()
    ax.set_title("dati.txt plot")
    ax.set_xlabel('variable')
    ax.set_ylabel('events in bin')
    ax.hist(data, bins=bin_edges, color="orange")
    plt.show()

    my_cost_func_2 = UnbinnedNLL(data, pdf)

    my_minuit_2 = Minuit(my_cost_func_2,
                         mu=sample_mean, sigma=sample_sigma)

    my_minuit_2.limits['sigma'] = (0, None)

    my_minuit_2.migrad()
    my_minuit_2.minos()
    display(my_minuit_2)
    return
