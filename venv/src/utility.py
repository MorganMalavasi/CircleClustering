import numpy as np

def histogram(theta, nbins=None, verb=True):
    if nbins is None:
        nbins = len(theta)
    # Return evenly spaced numbers over a specified interval.
    # start = 0
    # stop = 2*PI
    # nbins = Number of samples to generate
    bins = np.linspace(0,2*np.pi,nbins)
    h, b = np.histogram(theta, bins)
    return h, b

def averageOfList(lst):
    return sum(lst) / len(lst)