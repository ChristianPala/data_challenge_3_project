# Libraries:
from pathlib import Path

# networkx
import networkx as nx
import pandas as pd
from tqdm import tqdm


def main():
    # load the dataset for feature engineering:
    df = pd.read_csv(Path('data', 'online_sales_dataset_for_fe.csv'))

    # aggregate over the stock code, keeping a list of invoice numbers which contain the stock code:
    df_agg = df.groupby('StockCode').agg({'Invoice': lambda x: list(x)})

    # create a graph:
    G = nx.Graph()

    # add the nodes
    G.add_nodes_from(df_agg.index)

    # add the edges, the weight of the edge is the number of invoices in which the two stock codes
    # appear together
    for row in tqdm(df_agg.itertuples(), total=df_agg.shape[0]):
        for node in G.nodes:
            if node != row.Index:
                # get the intersection of the invoices:
                intersection = set(row.Invoice).intersection(set(df_agg.loc[node, 'Invoice']))
                # if the intersection is not empty:
                if intersection:
                    # add the edge:
                    G.add_edge(row.Index, node, weight=len(intersection))

    # save the graph:
    nx.write_gpickle(G, Path('feature_engineering', 'graph', 'saved_graphs', 'product_graph.gpickle'))


# Driver:
if __name__ == '__main__':
    main()
