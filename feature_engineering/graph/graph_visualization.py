# Libraries:

# Data manipulation:
from pathlib import Path

# Graph:
import networkx as nx

# Plotting:
from matplotlib import pyplot as plt

# Driver:
if __name__ == '__main__':

    G = nx.read_gpickle(Path('saved_graphs', 'customer_graph_country.gpickle'))

    G = nx.convert_node_labels_to_integers(G)

    subgraph = G.subgraph([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # plot the graph:
    plt.figure(figsize=(10, 10))
    nx.draw(subgraph, with_labels=True, node_size=1000, node_color='skyblue', edge_color='grey', width=2, font_size=12)
    plt.show()