# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path
# Graph:
import networkx as nx
# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split


# Functions:
def main(train_val_test_split: bool, train: bool, validation: bool, df_agg_path: Path) -> None:
    """
    Calculates the PageRank for each customer.
    @param train_val_test_split: bool: whether to split the data into train, validation and test sets.
    @param train: bool: whether to calculate the centrality measures for the training set.
    @param validation: bool: whether to calculate the centrality measures for the validation set.
    @param df_agg_path: path to the aggregated dataset.
    :return: None
    """
    df_agg = pd.read_csv(df_agg_path)

    if train_val_test_split:
        X_train, X_val, X_test, y_train, y_val, y_test = \
            train_validation_test_split(df_agg.drop('CustomerChurned', axis=1), df_agg['CustomerChurned'],
                                        validation=True)
        if train:
            graph_name: str = 'customer_graph_train.gpickle'
            c_id: pd.DataFrame = X_train['CustomerId']
            d_name: str = "customer_graph_train_centrality.csv"

        elif validation:
            graph_name: str = 'customer_graph_val.gpickle'
            c_id: pd.DataFrame = X_val['CustomerId']
            d_name: str = "customer_graph_val_centrality.csv"

        else:
            graph_name = 'customer_graph_test.gpickle'
            c_id = X_test['CustomerId']
            d_name: str = "customer_graph_test_centrality.csv"
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
    # closeness_centrality = nx.closeness_centrality(G)
    # print("Closeness centrality calculated.")
    # betweenness_centrality = nx.betweenness_centrality(G)
    # print("Betweenness centrality calculated.")
    # eigenvector_centrality = nx.eigenvector_centrality(G)
    # print("Eigenvector centrality calculated.")
    pagerank = nx.pagerank(G, weight='weight')
    print("PageRank calculated.")

    # save the measures to a dataframe:
    df_degree_centrality = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['DegreeCentrality'])

    # df_closeness_centrality = pd.DataFrame.from_dict(closeness_centrality, orient='index',
    #                                                  columns=['ClosenessCentrality'])
    # df_betweenness_centrality = pd.DataFrame.from_dict(betweenness_centrality, orient='index',
    #                                                    columns=['BetweennessCentrality'])
    # df_eigenvector_centrality = pd.DataFrame.from_dict(eigenvector_centrality, orient='index',
    #                                                   columns=['EigenvectorCentrality'])
    df_pagerank = pd.DataFrame.from_dict(pagerank, orient='index', columns=['PageRank'])

    df_centrality = pd.concat([df_degree_centrality, df_pagerank], axis=1)

    df_centrality.index.name = 'CustomerId'
    # save the pagerank to a csv file:
    df_centrality.to_csv(Path('..', '..', 'data', d_name))


# Driver:
if __name__ == '__main__':
    main(train_val_test_split=True, train=True, validation=False,
         df_agg_path=Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))
