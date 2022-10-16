# Libraries:

# Data manipulation:
import pandas as pd
from pathlib import Path

# Graph:
import networkx as nx

if __name__ == '__main__':
    # import the saved graph:
    G = nx.read_gpickle(Path('saved_graphs', 'customer_graph.gpickle'))

    # import the aggregated dataset:
    df_agg = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    # calculate the pagerank, dampening factors default is 0.85:
    pagerank = nx.pagerank(G, weight='weight')

    # save CustomerId and pagerank to a dataframe:
    df_pagerank = pd.DataFrame.from_dict(pagerank, orient='index', columns=['pagerank'])

    # add the CustomerId to the dataframe:
    df_pagerank['CustomerId'] = df_agg['CustomerId']

    # set the CustomerId as index:
    df_pagerank.set_index('CustomerId', inplace=True)

    # save the pagerank to a csv file:
    df_pagerank.to_csv(Path('..', '..', 'data', 'customer_pagerank.csv'))

