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


# circular space problem
def removeCircularSpace(hist):
    while True:
        pivot = findAnEmptySpace(hist)
        if pivot != -1:
            return rotateTheHistogram(hist, pivot)
        else:
            hist = removeSpace(hist)


def findAnEmptySpace(hist):
    for i in range(hist.shape[0]):
        if hist[i] == 0:
            return i
    return -1

def removeSpace(hist):
    minHist = min(hist)
    newHist = np.empty(hist.shape[0])
        
    for i in range(hist.shape[0]):
        newHist[i] = hist[i] - minHist
    
    return newHist

def rotateTheHistogram(hist, pivot):
    n = hist.shape[0]
    distance = n - pivot
    newHist = np.roll(hist, distance)
    return newHist
