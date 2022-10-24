# libraries
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import tsfel
from tqdm import tqdm
import warnings
# suppress warnings
warnings.filterwarnings("ignore")


def feature_extractor(data: pd.DataFrame, features, customer):
    # customers dataframes
    this_personDF = data.loc[data.CustomerId == customer].drop('CustomerId', axis=1)
    # returns ONE row for each the customer
    features_data = tsfel.time_series_features_extractor(features, this_personDF, verbose=0)
    return features_data


if __name__ == '__main__':

    agg = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))[["CustomerId", "CustomerChurned"]]

    # create a list of customers that have more than one purchase
    customers = agg[agg['NumberOfPurchases'] > 1]['CustomerId'].unique().tolist()

    # create a dataframe with the customers and the target variable
    y = y[y['CustomerId'].isin(customers)]

    # drop the customer_id from the labels:
    y.drop(columns=['CustomerId'], inplace=True)

    # import the timeseries dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))

    # create a column for the total spent:
    df['TotalSpent'] = df['Price'] * df['Quantity']

    # we cannot use the invoice date as it is a proxy for the target variable, we compute the recency:
    df = df[['CustomerId', 'TotalSpent', 'Quantity']]

    # cast the invoice date to datetime, then to int and divide by 10^9:
    # df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')
    # df['InvoiceDate'] = df['InvoiceDate'].astype(np.int64) // 10 ** 9

    # extract the features from the dataframe
    cfg = tsfel.get_features_by_domain(json_path='lib_files/features_mod.json')  # modified the json so that it doesnt
    # calculate useless features, like the ones specific for audio or EEG, etc..

    print('> execution started')

    # execute the feature extraction in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(feature_extractor, df, cfg, customer) for customer in tqdm(customers)]
        # wait for all the futures to finish
        results = [future.result() for future in tqdm(futures)]
        # catch exceptions:
        for future in futures:
            if future.exception() is not None:
                print(future.exception())
                # remove the future from the list
                futures.remove(future)
                # print the customer id that caused the exception
                print(customers[futures.index(future)])
        # concatenate the results as a dataframe
        X = pd.concat(results)

    print('> task mapped')

    # X['CustomerId'] = customers

    print(X.shape)
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_tsfel.csv'), index=False)
    y.to_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index=False)
