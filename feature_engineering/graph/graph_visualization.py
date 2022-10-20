# Libraries:

# Data manipulation:
from pathlib import Path

# Graph:
import networkx as nx

# Plotting:
from matplotlib import pyplot as plt

# Driver:
if __name__ == '__main__':

    G = nx.read_gpickle(Path('saved_graphs', 'product_graph.gpickle'))

    # plot the graph:
    nx.draw(G, with_labels=False, node_size=15)