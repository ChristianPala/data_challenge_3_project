# Libraries:
from multiprocessing import cpu_count
from pathlib import Path

import networkx as nx
import pandas as pd
from karateclub import DeepWalk


# Functions:
def main(graph_type_name: str, walk_length: int = 100, dimensions: int = 128) -> None:
    """
    Extracts deepwalk features from a graph and saves them as a csv file.
    @param graph_type_name: the name of the graph type to use.
    @param walk_length: the length of the random walk.
    @param dimensions: the number of dimensions of the embedding.
    :return: None: save the deepwalk features as a csv file.
    """
    # import the customer graph:
    G = nx.read_gpickle(Path('saved_graphs', f'{graph_type_name}_graph.gpickle'))

    # make sure the nodes are indexed as integers:
    G = nx.convert_node_labels_to_integers(G)

    # train a DeepWalk model:
    model = DeepWalk(walk_length=walk_length, dimensions=dimensions, workers=cpu_count() - 1)
    model.fit(G)
    embeddings = model.get_embedding()

    # save the embeddings to a dataframe:
    df_embeddings = pd.DataFrame(embeddings)

    # add the CustomerId to the dataframe:
    customer_ids = pd.read_csv(Path('data', 'online_sales_dataset_agg.csv'))['CustomerId']
    df_embeddings['CustomerId'] = customer_ids

    # set the CustomerId as index:
    df_embeddings.set_index('CustomerId', inplace=True)

    # save the embeddings to a csv file:
    df_embeddings.to_csv(Path('data', f'{graph_type_name}_deepwalk_embeddings.csv'))


if __name__ == '__main__':
    main('customer')
    main('customer_country')
    main('product')
