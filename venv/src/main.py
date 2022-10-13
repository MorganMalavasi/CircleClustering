import os
import numpy as np
import matplotlib.pyplot as plt
import engine
import data_generation
from rich.console import Console
import clustbench

os.environ["KMP_WARNINGS"] = "FALSE" 

plt.style.use('ggplot')
console = Console()

def main():
    
    # dataset = data_generation.createDatasets(6)
    # samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]

    data_path = os.path.join("clustering-data-v1-1.1.0")
    battery, dataset = "wut", "x2"
    benchmark = clustbench.load_dataset(battery, dataset, path=data_path)
    X = benchmark.data
    y_true = benchmark.labels[0]
    y_pred = engine.CircleClustering(X)+1

    print(y_true)
    print(y_pred)
    
    # engine.CircleClustering(samples, labels, n_dataset)

if __name__ == "__main__":
    main()
