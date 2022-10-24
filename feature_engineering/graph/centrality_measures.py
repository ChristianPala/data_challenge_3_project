# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path
# Graph:
import networkx as nx


# Functions:
def main(graph_type_name: str) -> None:
    """
    Extracts centrality measures from a graph and saves them as a csv file.
    @param graph_type_name: str: The name of the graph type to use.
    :return: None: Saves the centrality measures as a csv file.
    """

    graph_name = f'{graph_type_name}_graph.gpickle'
    d_name: str = f"{graph_type_name}_graph_centrality.csv"

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

    if graph_type_name == 'customer':
        df_centrality.index.name = 'CustomerId'
        # calculate the number of self-loops for each customer:
        self_loops = G.selfloop_edges()
        # add the self-loops to the dataframe:
        df_centrality['SelfLoops'] = [len([edge for edge in self_loops if edge[0] == node]) for node in
                                      df_centrality.index]

    if graph_type_name == 'product':
        df_centrality.index.name = 'StockCode'
        # calculate the number of self-loops for each product:
        self_loops = G.selfloop_edges()
        # add the self-loops to the dataframe:
        df_centrality['SelfLoops'] = [len([edge for edge in self_loops if edge[0] == node]) for node in
                                      df_centrality.index]

    # save the extracted features to a csv file:
    df_centrality.to_csv(Path('..', '..', 'data', d_name))


# Driver:
if __name__ == '__main__':
    main(graph_type_name='customer')
    main(graph_type_name='product')
