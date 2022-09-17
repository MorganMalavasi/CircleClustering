from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from scipy.stats import norm
from pandas import DataFrame
from sklearn import datasets
from sklearn.mixture import GaussianMixture

# constants
PI = np.pi
PI = np.float32(PI)

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

def mixtureOfGaussiansAutomatic(components, bins, theta):
    gmm = GaussianMixture(n_components = components)
    thetaReshaped = theta.reshape(-1,1)
    gmm.fit(thetaReshaped)

    labels = gmm.predict(thetaReshaped)
    
    plt.figure()
    plt.hist(theta, bins = bins, histtype='stepfilled', density=True, alpha=0.5)
    plt.xlim(0, 2*PI)

    f_axis = theta.copy().ravel()
    f_axis.sort()

    a = []
    for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
        a.append(weight*norm.pdf(f_axis, mean, np.sqrt(covar)).ravel())
        plt.plot(f_axis, a[-1])

    plt.plot(f_axis , np.array(a).sum(axis = 0), 'k-')
    plt.title("Gaussian mixture model")
    plt.xlabel("thetas")
    plt.ylabel("PDF")
    plt.tight_layout()

    plt.show()
    
    
    '''
    print(labels)
    plt.figure(figsize=(10,7))
    plt.xlabel("$points$")
    labels1 = []
    labels2 = []
    labels3 = []
    for i in range(theta.shape[0]):
        if labels[i] == 0:
            labels1.append(theta[i])
        if labels[i] == 1:
            labels2.append(theta[i])
        if labels[i] == 2:
            labels3.append(theta[i])

    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    labels3 = np.array(labels3)

    
    plt.scatter(labels1, [0.005] * len(labels1), color='r', s = 30, marker=2, label="cluster 1")
    plt.scatter(labels2, [0.005] * len(labels2), color='g', s = 30, marker=2, label="cluster 2")
    plt.scatter(labels3, [0.005] * len(labels3), color='b', s = 30, marker=2, label="cluster 3")

    plt.legend()
    plt.show()

    '''

    return (None, None, None)