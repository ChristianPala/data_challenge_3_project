# Libraries:

# Data manipulation:
from pathlib import Path

# Graph:
import networkx as nx

# Plotting:
from matplotlib import pyplot as plt

# Driver:
if __name__ == '__main__':
    # import the small graph:
    G = nx.read_gpickle(Path('saved_graphs', 'customer_graph_train_small.gpickle'))
    # plot the graph:
    nx.draw(G, with_labels=True)
    plt.show()