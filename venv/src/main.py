import os
import numpy as np
import matplotlib.pyplot as plt
import engine
import data_generation
from rich.console import Console

os.environ["KMP_WARNINGS"] = "FALSE" 

plt.style.use('ggplot')
console = Console()

def main():
    dataset = data_generation.createDatasets(6)
    samples, labels, n_dataset = dataset[0], dataset[1], dataset[2]
    engine.CircleClustering(samples, labels, n_dataset)

if __name__ == "__main__":
    main()
