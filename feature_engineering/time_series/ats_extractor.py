# libraries
# Multiprocessing:
from concurrent.futures import ProcessPoolExecutor
# Data manipulation:
from pathlib import Path
import pandas as pd
# Time series feature extraction library:
import tsfel
# Timing:
from tqdm import tqdm
# Warning Handling:
import warnings
# suppress warnings during feature extraction:
warnings.filterwarnings("ignore")


def feature_extractor(data: pd.DataFrame, features, customer) -> pd.DataFrame:
    """
    Extracts the time series features from the time series data.
    @param data: the time series data
    @param features: the features to extract
    @param customer: the customer id
    :return: pd.DataFrame: the extracted features
    """
    # customers dataframes
    this_customer = data.loc[data.CustomerId == customer].drop('CustomerId', axis=1)
    # returns ONE row for each customer df
    features_data = tsfel.time_series_features_extractor(features, this_customer, verbose=0)
    return features_data


if __name__ == '__main__':

    agg = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))[["CustomerId", "CustomerChurned"]]

    # create a list of customers that have more than one purchase
    customers = agg[agg['NumberOfPurchases'] > 1]['CustomerId'].unique().tolist()

    # create a dataframe with the customers and the target variable
    y = y[y['CustomerId'].isin(customers)]

    # import the timeseries dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))
    # convert the string date to a datetime
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate)
    df = df.sort_values(['CustomerId', 'InvoiceDate'])

    dates = df.copy()

    # calculate the difference between each date in days
    # (the .shift method will offset the rows, play around with this to understand how it works
    # - We apply this to every customer using a groupby
    df2 = dates.groupby("CustomerId").apply(lambda x: (x.InvoiceDate - x.InvoiceDate.shift(1)).dt.days).reset_index()
    df2 = df2.rename(columns={'InvoiceDate': 'AvgDays'})
    df2.index = df2.level_1

    # then join the result back on to the original dataframe
    dates = dates.join(df2['AvgDays'])

    # add the mean time to your groupby
    grouped = dates.groupby("CustomerId").agg({"AvgDays": "mean"})

    # rename columns per your original specification
    # grouped.columns = grouped.columns.get_level_values(0) + grouped.columns.get_level_values(1)
    dates = grouped.rename(columns={'AvgDays': 'avgTimeBetweenPurchases'})

    dates = dates[dates.index.isin(customers)]

    # create a column for the total spent:
    df['TotalSpent'] = df['Price'] * df['Quantity']

    for c in tqdm(dates.index.tolist()):
        if c in df['CustomerId'].tolist():
            df.loc[df['CustomerId'] == c, 'AvgDays'] = dates[dates.index == c].values

    # we cannot use the invoice date as it is a proxy for the target variable
    df = df[['CustomerId', 'TotalSpent', 'AvgDays']]

    # extract the features from the dataframe
    cfg = tsfel.get_features_by_domain(json_path='lib_files/features_mod.json')  # modified the json so that it $
    # does not calculate irrelevant features, like the ones specific for audio or EEG ... , it also ignores the
    # CustomerId for the extraction, the column is necessary for the grouping.

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

    # add the customer id as a column
    X['CustomerId'] = customers

    # remove all columns with NaN values:
    X = X.dropna(axis=1)

    # check the shape of the data
    print(X.shape)

    # check how many missing values we have for each feature:
    print(X.isna().sum())

    # save the features:
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_tsfel.csv'), index=False)
    y.to_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index=False)
