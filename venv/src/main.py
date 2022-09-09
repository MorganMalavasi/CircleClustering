import os
import numpy as np
import matplotlib.pyplot as plt
import cclustering_cpu as cc
import data_generation
import data_plot
import utility, histogram_clustering_hierarchical
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


dataset = data_generation.createDatasets(6)
samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]

'''CIRCLE CLUSTERING'''
numberOfSamplesInTheDataset = samples.shape[0]
theta = 2 * PI * np.random.rand(numberOfSamplesInTheDataset)
matrixOfWeights, S, C = cc.computing_weights(samples, theta, cosine = False)
theta = cc.loop(matrixOfWeights, theta, S, C, 0.001)

# //////////////////////////////////////////////////////////////////
# PLOTTING PCA
# data_plot.doPCA(samples, labels, n_dataset)

# PLOTTING THE THETA
# data_plot.plot_circle(theta)

hist, bins = utility.histogram(theta, nbins=128)

# PLOTTING THE SCATTER
# data_plot.plot_scatter(hist, bins, mode=2)
data_plot.plot_hist(hist, bins)
# //////////////////////////////////////////////////////////////////

'''
# smoothing 
# smooth values with average of ten values
# we are interested in the hist values because they represent the values to divide
hist_smoothed_weighted = smoothing_detection.smooth_weighted(hist)
data_plot.plot_scatter(hist_smoothed_weighted, bins, mode=2)
data_plot.plot_hist(hist_smoothed_weighted, bins)
'''

clusters, thetaLabels, centroids = histogram_clustering_hierarchical.hierarchical(hist, bins, samples, theta)
# print(clusters)

# PLOTTING THE THETA WITH COLOURS
# data_plot.plot_circle(theta, thetaLabels)
