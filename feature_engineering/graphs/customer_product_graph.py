# Libraries:
import networkx as nx
import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    # import the timeseries dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'), index_col='id')

    # create a graph:
    G = nx.Graph()

    # add CustomerIds as nodes:
    G.add_nodes_from(df['CustomerId'].unique())

    # add StockCodes as nodes:
    G.add_nodes_from(df['StockCode'].unique())

    # add edges:
    for _, row in df.iterrows():
        G.add_edge(row['CustomerId'], row['StockCode'])

    # print the number of nodes:
    print(f'Number of nodes: {G.number_of_nodes()}')

    # check if the graph is connected:
    print(f'Is the graph connected: {nx.is_connected(G)}')

    # print the number of connected components:
    print(f'Number of connected components: {nx.number_connected_components(G)}')






