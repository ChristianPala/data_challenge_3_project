# Libraries:
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# networkx
import networkx as nx


# Driver:
if __name__ == '__main__':

    # load the dataset for feature engineering:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))

    # aggregate over the stock code, keeping a list of invoice numbers which contain the stock code:
    df_agg = df.groupby('StockCode').agg({'Invoice': lambda x: list(x)})

    # create a graph:
    G = nx.Graph()

    # add the nodes
    G.add_nodes_from(df_agg.index)

    # add the edges, the weight of the edge is the number of invoices in which the two stock codes
    # appear together
    for i in tqdm(range(len(df_agg))):
        for j in range(len(df_agg)):
            if i != j:
                # get the invoice numbers of the two stock codes:
                invoices_i = df_agg.iloc[i, 0]
                invoices_j = df_agg.iloc[j, 0]
                # get the invoices in which the two stock codes appear together:
                invoices_shared = list(set(invoices_i).intersection(set(invoices_j)))
                # if there are invoices in which the two stock codes appear together, add an edge
                # with the weight of the number of invoices in which the two stock codes appear
                # together:
                if len(invoices_shared) > 0:
                    G.add_edge(df_agg.index[i], df_agg.index[j], weight=len(invoices_shared))

    # save the graph:
    nx.write_gpickle(G, Path('saved_graphs', 'product_graph.gpickle'))

