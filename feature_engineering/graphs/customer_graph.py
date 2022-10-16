# Reference: https://giotto-ai.github.io/gtda-docs/latest/notebooks/persistent_homology_graphs.html
# giotto-tda:
from gtda.graphs import GraphGeodesicDistance
# ask about this.

import numpy as np
# Libraries:
import pandas as pd
from pathlib import Path

# networkx
import networkx as nx

if __name__ == '__main__':
    # import the feature engineering dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))

    df['Weight'] = df['Quantity'] * df['Price']

    # Create a node of each unique customer:
    nodes = df['CustomerId'].unique()

    # Create a graph:
    G = nx.DiGraph()

    # Add the nodes:
    G.add_nodes_from(nodes)

    # Create a list of edges:
    edges = []

    # add edges if the customer has bought the same product as another customer, the weight is the
    # number of times they have bought the same product times the price of the product:
    for row in df.itertuples():
        # get the customers who have bought the same product via the stock code:
        customers = df[df['StockCode'] == df.loc[row.Index, 'StockCode']]['CustomerId'].unique()
        # create a list of edges with list comprehension:
        edges += [(df.loc[row.Index, 'CustomerId'], customer, df.loc[row.Index, 'Weight'])
                  for customer in customers if customer != df.loc[row.Index, 'CustomerId']]

    # add the edges to the graph:
    G.add_weighted_edges_from(edges)

    # calculate the pagerank, dampening factors default is 0.85:
    pagerank = nx.pagerank(G, weight='weight')

    # save the pagerank to a dataframe:
    df_pagerank = pd.DataFrame.from_dict(pagerank, orient='index', columns=['PageRank'])

    # save the pagerank to a csv file:
    df_pagerank.to_csv(Path('..', '..', 'data', 'customer_pagerank.csv'))