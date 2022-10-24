# auxiliary library to merge the nlp, time series and graph datasets

import pandas as pd
from pathlib import Path

if __name__ == '__main__':

    # import the RFM dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    y = df['CustomerChurned']

    # keep only the RFM features, change later to all the engineered features:
    X = df[['Recency', 'NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]

    # add the features from the time series analysis:
    # todo add the customer id to the time series dataset
    # import the time series dataset:
    df_ts = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_tsfel.csv'))
    # remove customer id's in the aggregated dataset that are not in the time series dataset:
    df = df[df['CustomerId'].isin(df_ts['CustomerId'])]
    # add the features from the time series dataset:
    X = pd.concat([X, df_ts.drop(['CustomerID'], axis=1)], axis=1)

    # add the clusters from the nlp analysis:
    # import the train nlp dataset:
    df_nlp = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_train.csv'))
    # import the test nlp dataset:
    df_nlp_test = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_test.csv'))
    # merge the train and test nlp datasets:
    df_nlp = pd.concat([df_nlp, df_nlp_test], axis=0)
    # remove the customer id's in the nlp dataset that are not in the aggregated dataset:
    df_nlp = df_nlp[df_nlp['CustomerId'].isin(df['CustomerId'])]
    # merge the nlp dataset with the aggregated dataset:
    df = df.merge(df_nlp, on='CustomerId')

    # add the features from the graph analysis:
    # todo: add the features from the graph analysis, save the fully aggregated dataset.

    # save the dataset:
    # todo: save the dataset with all the features

