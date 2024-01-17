import numpy as np
from scipy.stats import norm, expon
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from math import floor, ceil
from IPython.display import display


def cdf(bin_edges, N_signal, mu, sigma, N_background, tau):
    return N_signal * norm.cdf(bin_edges, mu, sigma) + N_background * expon.cdf(bin_edges, 0, tau)


def main():
    data = np.loadtxt("data/dati.txt")

    bin_content, bin_edges = np.histogram(data, bins=(floor(len(data) / 100)),
                                          range=(floor(min(data)), ceil(max(data))))

    sample_mean = np.mean(data)
    sample_sigma = np.std(data)

    fig, ax = plt.subplots()
    ax.set_title("dati.txt plot")
    ax.set_xlabel('variable')
    ax.set_ylabel('events in bin')
    ax.hist(data, bins=bin_edges, color="orange")
    plt.show()

    my_cost_func = ExtendedBinnedNLL(bin_content, bin_edges, cdf)

    N_events = sum(bin_content)

    my_minuit = Minuit(my_cost_func,
                       N_signal=N_events, mu=sample_mean, sigma=sample_sigma,
                       N_background=N_events, tau=1.)

    my_minuit.limits['N_signal', 'N_background', 'sigma', 'tau'] = (0, None)

    my_minuit.values["N_signal"] = 0
    my_minuit.fixed["N_signal", "mu", "sigma"] = True

    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    my_cost_func.mask = (bin_centres < 5) | (15 < bin_centres)

    my_minuit.migrad()
    my_minuit.minos()
    display(my_minuit)

    my_cost_func.mask = None
    my_minuit.fixed = False
    my_minuit.fixed["N_background", "tau"] = True
    my_minuit.values["N_signal"] = N_events - my_minuit.values["N_background"]

    my_minuit.migrad()
    my_minuit.minos()
    display(my_minuit)

    my_minuit.fixed = False
    my_minuit.migrad()
    my_minuit.minos()
    display(my_minuit)

    display(my_minuit.covariance.correlation())

    for key in my_minuit.parameters:  # parameters is a tuple containing the parameter names
        print('parameter ' + key + ': ' +
              str(my_minuit.values[key]) + ' +- ' +
              str(my_minuit.errors[key]))

    display(my_minuit.draw_mnmatrix())
    return
