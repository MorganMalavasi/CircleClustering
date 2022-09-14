import sys
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

def discretization(array, bins):
    return np.digitize(array, bins)

# circular space problem
def rotateHistogram(hist):
    # find min
    minHist = sys.maxsize
    index = 0
    for i in range(hist.shape[0]):
        if hist[i] < minHist:
            minHist = hist[i]
            index = i

    return rotate(hist, index)

def rotate(hist, pivot):
    n = hist.shape[0]
    distance = n - pivot
    newHist = np.roll(hist, distance)
    return newHist


# TODO -> find an heuristic for selecting the value of the percentage to remove
def removeLowPercentageOfNoise(hist):
    newHist = np.empty(hist.shape[0])
    
    maxHeight = max(hist)
    maxHeight_5_percent = maxHeight / 20
    for i in range(hist.shape[0]):
        if hist[i] < maxHeight_5_percent:
            newHist[i] = 0
        else:
            newHist[i] = hist[i]
    
    return newHist
    

# old version
def removeCircularSpace(hist):
    while True:
        pivot = findAnEmptySpace(hist)
        if pivot != -1:
            return rotate(hist, pivot)
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

# smoothing
def smooth_weighted(values):
    """
    Compute the smoothing of a line of values
    Given a line of "values", it is applied an averaging filter for smoothing it

    Parameters
    ----------
    values : ndarray
        1D NumPy array of float dtype representing n-dimensional points on a chart
    smoothing_index : int
        value representing the size of the window for smoothing
        default = 10

    Returns
    -------
    output : ndarray 
        line of "values" smoothed
    """
    output = np.empty([values.shape[0]])

    # define the weight for the gaussian
    smoothing_index = 7
    gaussianWeights = np.array([0.25, 1, 2, 4, 2, 1, 0.25])

    for i in range(values.shape[0]):
        sum = 0.0
        count = 0
        for j in range(smoothing_index):
            x = j - int(smoothing_index/2)
            if (i+x)>=0 and (i+x)<values.shape[0]:
                sum = sum + values[i+x] * gaussianWeights[j]
                count += 1

        output[i] = sum / count

    return output