# auxiliary library to merge the nlp, time series and graph datasets
# Libraries:
# Data manipulation:
from pathlib import Path

import pandas as pd


def main():
    # import the RFM dataset:
    df = pd.read_csv(Path('data', 'online_sales_dataset_agg.csv'))

    y = df['CustomerChurned']

    # remove the target variable and the description from the dataset:
    df.drop(['CustomerChurned', 'Description'], axis=1, inplace=True)

    # add the features from the time series analysis:
    # import the time series dataset:
    df_ts = pd.read_csv(Path('data', 'online_sales_dataset_tsfel_for_fs.csv'))

    # check how many customers are in the aggregated dataset and not in the time series dataset:
    print(f"Number of customers in the aggregated dataset and not in the time series dataset: "
          f"{df.shape[0] - df_ts.shape[0]}")
    # Since we are losing few customers, we will proceed with the merge:

    # remove customer id's in the aggregated dataset that are not in the time series dataset:
    df = df[df['CustomerId'].isin(df_ts['CustomerId'])]

    # merge the time series features with the aggregated dataset:
    df = df.merge(df_ts, how='left', on='CustomerId')

    # import the nlp dataset:
    df_nlp = pd.read_csv(Path('data', 'online_sales_dataset_nlp_for_fs.csv'), index_col=0)
    # merge the nlp dataset with the aggregated dataset:
    df = df.merge(df_nlp, how='left', left_on='CustomerId', right_on='CustomerId')

    # import the graph dataset:
    df_graph = pd.read_csv(Path('data', 'online_sales_dataset_graph_for_fs.csv'), index_col=0)
    # merge the graph dataset with the aggregated dataset:
    df = df.merge(df_graph, how='left', left_on='CustomerId', right_on='CustomerId')

    # if there are missing values, raise an exception:
    if df.isnull().sum().sum() > 0:
        raise Exception('There are missing values in the dataset.')

    # save the dataset, the first row is the header:
    df.to_csv(Path('data', 'online_sales_dataset_for_fs.csv'), index=False)
    # print the number of features:
    print(f"Number of features: {df.shape[1] - 1}")


if __name__ == '__main__':
    main()
