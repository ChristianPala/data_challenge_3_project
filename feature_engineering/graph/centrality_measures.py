# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path
# Graph:
import networkx as nx
# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split


# Functions:
def main(train_test_split: bool, train: bool, df_agg_path: Path) -> None:
    """
    Calculates the PageRank for each customer.
    @param train_test_split: bool: whether to split the data into train and test sets.
    @param train: bool: True if we want to calculate the PageRank for the training set, False for the testing set.
    @param df_agg_path: path to the aggregated dataset.
    :return: None
    """
    df_agg = pd.read_csv(df_agg_path)

    if train_test_split:
        X_train, X_test, y_train, y_test = \
            train_validation_test_split(df_agg.drop('CustomerChurned', axis=1), df_agg['CustomerChurned'])
        if train:
            graph_name: str = 'customer_train_graph.gpickle'
            c_id: pd.DataFrame = X_train['CustomerId']
            d_name: str = "customer_train_graph_centrality.csv"
        else:
            graph_name = 'customer_test_graph.gpickle'
            c_id = X_test['CustomerId']
            d_name: str = "customer_test_graph_centrality.csv"
    else:
        graph_name = 'customer_graph.gpickle'
        c_id = df_agg['CustomerId']
        d_name: str = "customer_graph_centrality.csv"

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

    # add the CustomerId to the dataframe:
    df_centrality['CustomerId'] = c_id

    # set the CustomerId as index:
    df_pagerank.set_index('CustomerId', inplace=True)

    # save the pagerank to a csv file:
    df_pagerank.to_csv(Path('..', '..', 'data', d_name))


# Driver:
if __name__ == '__main__':
    main(train_test_split=True, train=True,
         df_agg_path=Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))
