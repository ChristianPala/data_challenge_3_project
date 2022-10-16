# Reference: https://giotto-ai.github.io/gtda-docs/latest/notebooks/persistent_homology_graphs.html
# giotto-tda:
# todo: ask about this.

# Libraries:
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# networkx
import networkx as nx

if __name__ == '__main__':
    # import the feature engineering dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))
    df_agg_ids = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))['CustomerId']

    # remove ids in df not in df_agg_ids due to aggregated cancelling order possible mismatches:
    df = df[df['CustomerId'].isin(df_agg_ids)]

    # create a column with the future weights of the graph, round to 2 decimals:
    df['Weight'] = (df['Quantity'] * df['Price']).round(2)

    # Create a graph:
    G = nx.DiGraph()

    # Create a node of each unique customer:
    nodes = df['CustomerId'].unique()

    # Add the nodes:
    G.add_nodes_from(nodes)

    # aggregate the dataset by customerId, save the stock codes as a list, the weights as a tuple of
    # stock code and weight:
    df_agg = df.groupby('CustomerId')\
        .agg({'StockCode': lambda x: list(x),
              'Weight': lambda x: list(zip(df.loc[x.index, 'StockCode'], x))})

    # add an edge to the graph if the stock code is in the list of stock codes of the other customer,
    # the weight of the edge is the sum of the weights of the pzsroducts that the two customers have
    # in common, add tdqm to show progress:

    for i in tqdm(range(len(df_agg))):
        for j in range(len(df_agg)):
            if i != j:
                # get the stock codes of the two customers:
                stock_codes_i = df_agg.iloc[i, 0]
                stock_codes_j = df_agg.iloc[j, 0]
                # get the weights of the two customers:
                weights_i = df_agg.iloc[i, 1]
                weights_j = df_agg.iloc[j, 1]
                # get the common stock codes:
                common_stock_codes = list(set(stock_codes_i) & set(stock_codes_j))
                # if there are common stock codes, add two edges with the sum of the weights of the
                # common stock codes for each customer:
                if len(common_stock_codes) > 0:
                    # get the weights of the common stock codes for each customer:
                    weights_i = [weight for stock_code, weight in weights_i if stock_code in common_stock_codes]
                    weights_j = [weight for stock_code, weight in weights_j if stock_code in common_stock_codes]
                    # add the edges:
                    G.add_edge(df_agg.index[i], df_agg.index[j], weight=sum(weights_i))
                    G.add_edge(df_agg.index[j], df_agg.index[i], weight=sum(weights_j))

    # save the graph:
    nx.write_gpickle(G, Path('saved_graphs', 'customer_graph.gpickle'))

