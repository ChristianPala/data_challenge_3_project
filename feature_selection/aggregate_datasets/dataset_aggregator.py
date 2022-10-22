# auxiliary library to merge the nlp, time series and graph datasets

import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    # import the nlp dataset:
    df_nlp = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp.csv'))

    # import the time series dataset:
    df_ts = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_tsfel.csv'))

    # import the graph dataset:
    df_graph_product = pd.read_csv(Path('..', '..', 'data', 'product_graph_centrality.csv'))
    df_graph_customer = pd.read_csv(Path('..', '..', 'data', 'customer_graph_centrality.csv'))

    # merge the datasets:
    df = pd.merge(df_nlp, df_ts, on='CustomerId')
    df = pd.merge(df, df_graph_customer, on='CustomerId')

    #





    # save the merged dataset:
    df.to_csv(Path('..', '..', 'data', 'online_sales_dataset_merged.csv'))