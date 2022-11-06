# Libraries
# Data manipulation:
import pandas as pd
from pathlib import Path

# Graph:
import networkx as nx

# Timing:
from tqdm import tqdm

def main():

    # import the aggregated dataset:
    df_agg = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'), index_col=0)

    # aggregate the dataset by customerId, save the country as a list:
    df_agg = df_agg.groupby('CustomerId').agg({'Country': lambda x: list(x)})

    # create a simple graph:
    G = nx.Graph()

    # add the nodes as the customer ids:
    G.add_nodes_from(df_agg.index)

    # add the edges:
    for row in tqdm(df_agg.itertuples(), total=df_agg.shape[0]):
        for node in G.nodes:
            if node != row.Index:
                # if one of the countries is in the list of countries of the other customer:
                if set(row.Country).intersection(set(df_agg.loc[node, 'Country'])):
                    # add the edge:
                    G.add_edge(row.Index, node)

    # save the graph:
    nx.write_gpickle(G, Path('saved_graphs', 'customer_country_graph.gpickle'))

if __name__ == '__main__':
    main()
