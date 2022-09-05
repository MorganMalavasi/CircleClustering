import os
import numpy as np
import matplotlib.pyplot as plt
import cclustering_cpu as cc
import data_generation
import data_plot
import smoothing_detection, utility, histogram_clustering_hierarchical
from utility import averageOfList
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from multiprocessing import Process

os.environ["KMP_WARNINGS"] = "FALSE" 

plt.style.use('ggplot')
console = Console()

# constants
PI = np.pi
PI = np.float32(PI)


dataset = data_generation.createDatasets(2)
samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]

#  if j == 0:
# console.print("samples = {0}, centroids = {1}".format(samples.shape[0], max(labels) + 1), )

'''CIRCLE CLUSTERING'''
numberOfSamplesInTheDataset = samples.shape[0]
theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
matrixOfWeights, S, C = cc.computing_weights(samples, theta, cosine = False)
theta = cc.loop(matrixOfWeights, theta, S, C, 0.001)


# data_plot.doPCA(samples, labels, n_dataset)
# data_plot.plot_circle(theta)
hist, bins = utility.histogram(theta, nbins=128)
# Plot the histogram
# data_plot.plot_scatter(hist, bins, mode=2)
# data_plot.plot_hist(hist, bins)
# remove 10% of the noise in the data
maxHeight = max(hist)
maxHeight_5_percent = maxHeight / 20 
for i in range(hist.shape[0]):
    if hist[i] < maxHeight_5_percent:
        hist[i] = 0
# data_plot.plot_scatter(hist, bins, mode=2)
# data_plot.plot_hist(hist, bins)

'''
# smoothing 
# smooth values with average of ten values
# we are interested in the hist values because they represent the values to divide
hist_smoothed_weighted = smoothing_detection.smooth_weighted(hist)
data_plot.plot_scatter(hist_smoothed_weighted, bins, mode=2)
data_plot.plot_hist(hist_smoothed_weighted, bins)
'''
# new algorithm for counting the number of clusters in an histogram of densities
clusters = histogram_clustering_hierarchical.getClustersFromHistogram(hist, bins)
thetaLabels = histogram_clustering_hierarchical.labelTheSamples(samples, theta, clusters, bins)
centroids = histogram_clustering_hierarchical.centroidsFinder(samples, thetaLabels)
# data_plot.plot_circle(theta, thetaLabels)

"""

# computing histogram values in 512 bins
hist, bins = data_plot.histogram(theta, nbins=512)
data_plot.plot_hist(hist, bins, mode=2)


# smoothing 
# smooth values with average of ten values
# we are interested in the hist values because they represent the values to divide
hist_smoothed = smoothing_detection.smooth(hist)
data_plot.plot_hist(hist_smoothed, bins, mode=2)

hist_smoothed_weighted = smoothing_detection.smooth_weighted(hist)
data_plot.plot_hist(hist_smoothed_weighted, bins, mode=2)

# detection
# detect how many wells there are
# 1) in the real 
nClusters, weights = smoothing_detection.simple_detection(hist)
#print("there are {0} clusters".format(nClusters))

# 2) smoothed
nClusters, weights = smoothing_detection.simple_detection(hist_smoothed)
#print("there are {0} smoothed clusters".format(nClusters))

# 2) smoothed with weights
nClusters, weights = smoothing_detection.simple_detection(hist_smoothed_weighted)
#print("there are {0} smoothed clusters with weights".format(nClusters))


"""

