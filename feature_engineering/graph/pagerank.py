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
            pagerank_id: pd.DataFrame = X_train['CustomerId']
            pagerank_name: str = "customer_train_pagerank.csv"
        else:
            graph_name = 'customer_test_graph.gpickle'
            pagerank_id = X_test['CustomerId']
            pagerank_name: str = "customer_test_pagerank.csv"
    else:
        graph_name = 'customer_graph.gpickle'
        pagerank_id = df_agg['CustomerId']
        pagerank_name: str = "customer_pagerank.csv"

    # import the saved graph:
    G = nx.read_gpickle(Path('saved_graphs', graph_name))

    pagerank = nx.pagerank(G, weight='weight')

    # save the pagerank to a dataframe:
    df_pagerank = pd.DataFrame.from_dict(pagerank, orient='index', columns=['PageRank'])

    # add the CustomerId to the dataframe:
    df_pagerank['CustomerId'] = pagerank_id

    # set the CustomerId as index:
    df_pagerank.set_index('CustomerId', inplace=True)

    # save the pagerank to a csv file:
    df_pagerank.to_csv(Path('..', '..', 'data', pagerank_name))


# Driver:
if __name__ == '__main__':
    main(train_test_split=True, train=True,
         df_agg_path=Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))
