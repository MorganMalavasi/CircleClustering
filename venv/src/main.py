import os
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
import engine
import clustbench
import sklearn.metrics as metrics
import sklearn.cluster as clustering_sklearn
import genieclust

os.environ["KMP_WARNINGS"] = "FALSE" 

plt.style.use('ggplot')
console = Console()

def main():
    
    # dataset = data_generation.createDatasets(6)
    # samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]

    """
        Here we build a mega procedure for testing our clustering algorithms.
        We will benchmark and compare the following algorithms:
            - CircleClustering
            - ...

        For the testing we will use the framework 'Clustering Benchmarks' by Marek Gagolewski
        with the batteries from the big dataset always from Marek G.

        In paricular the DB is composed of 9 main groups of dataset (batteries).
        ['fcps', 'g2mg', 'graves', 'h2mg', 'mnist', 'other', 'sipu', 'uci', 'wut']
        Most important, each group is composed of a big number of different datasets
        
        For comparing the differences and enstablish the best algorithms among all
        we will use some external validity indeces. 
        [...]

    """
    data_path = os.path.join("clustering-data-v1-1.1.0")
    
    batteries_names = clustbench.get_battery_names(path=data_path)
    # loop on all the groups of datasets
    for eachBatteryName in batteries_names:
        # loop on each dataset in the battery
        battery = clustbench.get_dataset_names(eachBatteryName, path=data_path)
        print(eachBatteryName)
        for eachDatasetName in battery:
            print("- {0}".format(eachDatasetName))
            benchmark = clustbench.load_dataset(eachBatteryName, eachDatasetName, path=data_path)
            X = benchmark.data
            y_true = benchmark.labels[0]
            correct_number_of_clusters = max(y_true)
            print("Dataset size {0}".format(len(X)))

            # Circle Clustering
            y_pred_circle_clustering = engine.CircleClustering(X) + 1

            # k-means
            y_pred_k_means = clustering_sklearn.KMeans(correct_number_of_clusters).fit(X).labels_ + 1

            # affinity propagation
            y_pred_affinity_propagation = clustering_sklearn.AffinityPropagation().fit(X).labels_ + 1

            # mean shift
            y_pred_mean_shift = clustering_sklearn.MeanShift().fit(X).labels_

            # genie
            y_pred_genie = genieclust.Genie(n_clusters=correct_number_of_clusters).fit_predict(X) + 1

            # hierarchical clustering
            # - ward
            # - average linkage
            # - complete linkage
            # - ward linkage

            # dbscan

            # optics

            # birch

            # spectral clustering

            # dbscan

            # optics

        break






    
    '''
    battery, dataset = "wut", "x2"
    benchmark = clustbench.load_dataset(battery, dataset, path=data_path)
    X = benchmark.data
    y_true = benchmark.labels[0]
    y_pred = engine.CircleClustering(X)+1

    
    # y_pred[-1] = 3
    for i in range(len(y_true)):
        if y_true[i] == 3:
            y_true[i] = 2
        
    y_pred[-1] = 3

    print(y_true)
    print(y_pred)

    print(len(y_true))
    print(len(y_pred))

    # print(clustbench.get_score(y_true, y_pred))
    print(metrics.adjusted_rand_score(y_true, y_pred))
    
    # engine.CircleClustering(samples, labels, n_dataset)

    '''

if __name__ == "__main__":
    main()
