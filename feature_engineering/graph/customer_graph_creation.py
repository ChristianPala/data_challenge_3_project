# Libraries:
from pathlib import Path

# Graph:
import networkx as nx
import pandas as pd
from tqdm import tqdm


def main():
    # import the feature engineering dataset:
    df = pd.read_csv(Path('data', 'online_sales_dataset_for_fe.csv'))

    # create a column with the future weights of the graph, round to 2 decimals:
    df['Weight'] = (df['Quantity'] * df['Price']).round(2)

    # Create a graph:
    G = nx.DiGraph()

    # aggregate the dataset by customerId, save the stock codes as a list, the weights as a tuple of
    # stock code and weight:

    df_agg = df.groupby('CustomerId').agg({'StockCode': lambda x: list(x),
                                           'Weight': lambda x: list(zip(df.loc[x.index, 'StockCode'], x)),
                                           'Country': lambda x: x.value_counts().index[0]})

    # create the graph:
    # Add the nodes customer ids as nodes, the country as attribute
    G.add_nodes_from(df_agg.index, country=df_agg['Country'])

    # add an edge to the graph if the stock code is in the list of stock codes of the other customer,
    # the weight of the edge is the sum of the weights of the products that the two customers have
    # in common:
    for row in tqdm(df_agg.itertuples(), total=df_agg.shape[0]):
        for node in G.nodes:
            if node != row.Index:
                # get the intersection of the stock codes:
                intersection = set(row.StockCode).intersection(set(df_agg.loc[node, 'StockCode']))
                # if the intersection is not empty:
                if intersection:
                    # get the weights of the intersection:
                    weights = [weight for stock_code, weight in df_agg.loc[node, 'Weight']
                               if stock_code in intersection]
                    # add the edge:
                    G.add_edge(row.Index, node, weight=sum(weights))

    # save the graph:
    nx.write_gpickle(G, Path('saved_graphs', 'customer_graph.gpickle'))


if __name__ == '__main__':
    main()
