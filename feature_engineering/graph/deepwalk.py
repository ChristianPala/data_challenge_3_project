# Libraries:
import pandas as pd
from pathlib import Path
from karateclub import DeepWalk
import networkx as nx
from multiprocessing import cpu_count

if __name__ == '__main__':
    # import the customer graph:
    G = nx.read_gpickle(Path('saved_graphs', 'customer_graph.gpickle'))

    # make sure the nodes are indexed as integers:
    # G = nx.convert_node_labels_to_integers(G)

    # train a DeepWalk model:
    model = DeepWalk(walk_length=100, dimensions=128, workers=cpu_count() - 1)
    model.fit(G)
    embeddings = model.get_embedding()

    # save the embeddings to a dataframe:
    df_embeddings = pd.DataFrame(embeddings)

    # add the CustomerId to the dataframe:
    customer_ids = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))['CustomerId']
    df_embeddings['CustomerId'] = customer_ids

    # set the CustomerId as index:
    df_embeddings.set_index('CustomerId', inplace=True)

    # save the embeddings to a csv file:
    df_embeddings.to_csv(Path('..', '..', 'data', 'customer_deepwalk_embeddings.csv'))

    #  Doing something wrong or the feature is terrible.

