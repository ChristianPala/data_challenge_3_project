# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path
# Graph:
import networkx as nx
# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split


# Functions:
def main(graph_type: str) -> None:
    """
    Calculates the PageRank for each customer.
    @param graph_type: The type of graph to use.
    :return: None
    """

    graph_name = f'{graph_type}_graph.gpickle'
    d_name: str = f"{graph_type}_graph_centrality.csv"

    # import the saved graph:
    G = nx.read_gpickle(Path('saved_graphs', graph_name))

    # Calculate the centrality measures:
    print("Calculating the centrality measures...")
    degree_centrality = nx.degree_centrality(G)
    print("Degree centrality calculated.")
    closeness_centrality = nx.closeness_centrality(G)
    print("Closeness centrality calculated.")
    betweenness_centrality = nx.betweenness_centrality(G)
    print("Betweenness centrality calculated.")
    eigenvector_centrality = nx.eigenvector_centrality(G)
    print("Eigenvector centrality calculated.")
    pagerank = nx.pagerank(G, weight='weight')
    print("PageRank calculated.")

    # save the measures to a dataframe:
    df_degree_centrality = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['DegreeCentrality'])

    df_closeness_centrality = pd.DataFrame.from_dict(closeness_centrality, orient='index',
                                                     columns=['ClosenessCentrality'])
    df_betweenness_centrality = pd.DataFrame.from_dict(betweenness_centrality, orient='index',
                                                       columns=['BetweennessCentrality'])
    df_eigenvector_centrality = pd.DataFrame.from_dict(eigenvector_centrality, orient='index',
                                                       columns=['EigenvectorCentrality'])
    df_pagerank = pd.DataFrame.from_dict(pagerank, orient='index', columns=['PageRank'])

    df_centrality = pd.concat([df_degree_centrality, df_closeness_centrality, df_betweenness_centrality,
                               df_eigenvector_centrality, df_pagerank], axis=1)

    df_centrality.index.name = 'CustomerId'
    # save the extracted features to a csv file:
    df_centrality.to_csv(Path('..', '..', 'data', d_name))


# Driver:
if __name__ == '__main__':
    main(graph_type='product')