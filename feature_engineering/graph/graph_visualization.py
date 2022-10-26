# Libraries:

# Data manipulation:
from pathlib import Path

# Graph:
import networkx as nx

# Plotting:
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Driver:
if __name__ == '__main__':

    G = nx.read_gpickle(Path('saved_graphs', 'customer_country_graph.gpickle'))

    # print the number of nodes and edges:
    print(f'Number of nodes: {G.number_of_nodes()}')
    print(f'Number of edges: {G.number_of_edges()}')

    # print the number of connected components:
    print(f'Number of connected components: {nx.number_connected_components(G)}')

    # if the graph labels are not integers, convert them to integers:
    if not all(isinstance(node, int) for node in G.nodes):
        G = nx.convert_node_labels_to_integers(G)

    # select a subset of the graph:
    sample_size = 100
    G = G.subgraph(list(G.nodes)[:sample_size])

    # plot the graph:
    plt.figure(figsize=(10, 10))
    nx.draw(G, node_size=100, cmap=plt.cm.Blues, font_size=8)
    plt.title('Customer Country Graph')
    plt.show()
