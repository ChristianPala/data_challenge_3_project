# Libraries:
import networkx as nx
import pandas as pd
from pathlib import Path


from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':
    # import the timeseries dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))

    # group by customer:
    df_c = df.groupby('CustomerId').agg({'StockId': lambda x: list(x), 'Country': lambda x: x.value_counts().index[0]})
    df_s = df.groupby('StockId').agg({'Price': 'mean'})
    # create a graph:
    G = nx.Graph()

    # add CustomerIds as nodes with blue color:
    G.add_nodes_from(df_c['CustomerId'], color='blue', size=5, country=df['Country'])
    G.add_nodes_from(df_s['StockId'], color='red', size=5, price=df['Price'])

    # add edges if a customer bought a product, the weight is the number of products




    # print the number of nodes:
    print(f'Number of nodes: {G.number_of_nodes()}')

    # check if the graph is connected:
    print(f'Is the graph connected: {nx.is_connected(G)}')

    # print the number of connected components:
    print(f'Number of connected components: {nx.number_connected_components(G)}')

    # print the number of nodes:
    print(f'Number of nodes: {G.number_of_nodes()}')

    # check if the graph is connected:
    print(f'Is the graph connected: {nx.is_connected(G)}')

    # print the number of connected components:
    print(f'Number of connected components: {nx.number_connected_components(G)}')

    # print the number of edges:
    print(f'Number of edges: {G.number_of_edges()}')

    # save the graph with pylab:
    nx.draw(G, with_labels=True, node_size=5, font_size=5)
    plt.savefig(Path('..', '..', 'plots', 'customer_product_graph.png'))
