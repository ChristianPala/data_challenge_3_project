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

    # create a graph:
    G = nx.Graph()

    # add CustomerIds as nodes:
    G.add_nodes_from(df['CustomerId'].unique())

    # add country as an attribute to the customer nodes:
    for customer_id in df['CustomerId'].unique():
        G.nodes[customer_id]['country'] = df[df['CustomerId'] == customer_id]['Country'].value_counts().index[0]

    # add StockCodes as nodes:
    G.add_nodes_from(df['StockCode'].unique())

    # add edges for each customer-product pair, the quantity as the weight:
    for customer_id in df['CustomerId'].unique():
        for stock_code in df[df['CustomerId'] == customer_id]['StockCode'].unique():
            G.add_edge(customer_id, stock_code, weight=
            df[(df['CustomerId'] == customer_id) & (df['StockCode'] == stock_code)]['Quantity'].sum())

    # add the number of times a customer bought a product as an attribute to the edge:
    for _, row in df.iterrows():
        G[row['CustomerId']][row['StockCode']]['count'] = G[row['CustomerId']][row['StockCode']]\
                                                              .get('count', 0) + 1

    # add countries as attributes to the customer nodes:
    for customer_id in df['CustomerId'].unique():
        G.nodes[customer_id]['country'] = df[df['CustomerId'] == customer_id]['Country'].value_counts().index[0]

    # print the number of nodes:
    print(f'Number of nodes: {G.number_of_nodes()}')

    # check if the graph is connected:
    print(f'Is the graph connected: {nx.is_connected(G)}')

    # print the number of connected components:
    print(f'Number of connected components: {nx.number_connected_components(G)}')

    nx.draw(G)
    plt.show()

    # print the number of nodes:
    print(f'Number of nodes: {G.number_of_nodes()}')

    # check if the graph is connected:
    print(f'Is the graph connected: {nx.is_connected(G)}')

    # print the number of connected components:
    print(f'Number of connected components: {nx.number_connected_components(G)}')

    # print the number of edges:
    print(f'Number of edges: {G.number_of_edges()}')

    # print the number of self loops:
    print(f'Number of self loops: {G.number_of_selfloops()}')

    nx.draw(G)
    plt.show()

    # save the graph:
    nx.write_gpickle(G, Path('..', 'data', 'customer_product_graph.gpickle'))



