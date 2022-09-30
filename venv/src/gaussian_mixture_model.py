import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from pandas import DataFrame
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from data_plot import drawMixtureOfGaussians


######################### IMPLEMENTATION OF MIXTURE OF GAUSSIANS MANUALLY ###########################
def mixtureOfGaussiansManual(components, bins, theta):
    
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

    k = components
    weights = np.ones((k)) / k 
    means = np.random.choice(theta, k)
    variances = np.random.random_sample(size = k)
    # print(means, variances)

    eps = 1e-8
    steps = 100
    for step in range(100):

        likelihood = []
        for j in range(k):
            likelihood.append(pdf(theta, means[j], np.sqrt(variances[j])))
        likelihood = np.array(likelihood)

        b = []
        # maximization step
        for j in range(k):
            # use the current values for the parameters to evaluate the posterior
            # probabilities of the data to have been generanted by each gaussian    
            b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0)+eps))

            # updage mean and variance
            means[j] = np.sum(b[j] * theta) / (np.sum(b[j]+eps))
            variances[j] = np.sum(b[j] * np.square(theta - means[j])) / (np.sum(b[j]+eps))

            # update the weights
            weights[j] = np.mean(b[j])

    plt.figure(figsize=(10, 6))
    axes = plt.gca()
    plt.xlabel("$samples$")
    plt.ylabel("pdf")
    plt.title("Gaussian mixture model")
    plt.scatter(theta, [-0.05] * len(theta), color='navy', s=30, marker=2, label="Train data")

    for i in range(components):
        plt.plot(bins, pdf(bins, means[i], variances[i]), color=colors[i], label="Cluster {0}".format(i+1))
    
    plt.legend(loc='upper left')
    
    # plt.savefig("img_{0:02d}".format(step), bbox_inches='tight')
    plt.show()
    return 

def pdf(data, mean: float, variance: float):
  # A normal continuous random variable.
  s1 = 1/(np.sqrt(2*np.pi*variance))
  s2 = np.exp(-(np.square(data - mean)/(2*variance)))
  return s1 * s2

######################### IMPLEMENTATION OF MIXTURE OF GAUSSIANS WITH SKLEARN ########################

def mixtureOfGaussiansAutomatic(k, bins, theta):
    # evaluate the best model for the gaussian mixture using bic 
    # we check in the neighbours of the k found
    lowest_bic = np.infty
    bic = []
    n_components_range_lower = range(k-2, k)
    n_components_range_higher = range(k, k+3)
    n_components_range = chain(n_components_range_lower, n_components_range_higher)

    print("----------------")
    for i in n_components_range:
        if i < 0:
            continue
        gmm = GaussianMixture(n_components = i)
        thetaReshaped = theta.reshape(-1,1)
        gmm.fit(thetaReshaped)
        bic.append(gmm.bic(thetaReshaped))
        print("Nr. components : {0}, bic = {1}".format(i, bic[-1]))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
    print("----------------")


    drawMixtureOfGaussians(theta, bins, best_gmm)

    labels = best_gmm.predict(thetaReshaped)
    return labels