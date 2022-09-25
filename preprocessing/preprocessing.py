# Libraries:
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


if __name__ == '__main__':
    # data file_paths
    f_09 = Path('../data/online_sales_2009_2010_dataset.csv')
    f_10 = Path('../data/online_sales_2010_2011_dataset.csv')
    # Load data
    df_09 = pd.read_csv(f_09, sep=';')
    df_10 = pd.read_csv(f_10, sep=';')
    # merge the two datasets for preprocessing:
    pd.concat([df_09, df_10], axis=0).to_csv('../data/online_sales_dataset.csv', index=False)
    df = pd.read_csv('../data/online_sales_dataset.csv')
    # First look
    print("First look at the data:")
    print(df.head())
    print(df.info())
    # check the columns types:
    print(df.dtypes)
    # need to change the dates to timestamps
    # drop the cancelled invoices (for now, then we may want to use them later in our analysis)
    # check unique values in each column:
    for col in df.columns:
        print(col, df[col].unique().size)
    # check the number of missing values in each column:
    print(df.isnull().sum())
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # convert InvoiceDate to datetime:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')

    # check the dataset size:
    size = df.shape[0]

    # check for duplicates after the merge of the two csv files:
    df = df.drop_duplicates(keep='first')

    # check if duplicates were removed:
    print(f'Number of duplicates removed: {df.shape[0] - size}')

    # drop all missing values, for now while CB is working on imputing:
    # df.dropna(inplace=True)

    # print unique Customer ID's:
    print(f"We have: {df['Customer ID'].unique().size} unique customers")

    # find the maximum time elapsed between invoices from the same customer:
    print(f"The longest repurchase in the dataset happened after: "
          f"{df.groupby('Customer ID')['InvoiceDate'].diff().max()} ")
    # not very promising, we need to check how many "false positives" for a given time window there are then,
    # to decide on the churn threshold.

    thresholds = {31, 62, 93, 186, 365}

    for t in thresholds:
        print(f"False positives after {t} days:"
              f"{(df.groupby('Customer ID')['InvoiceDate'].max() - df.groupby('Customer ID')['InvoiceDate'].min() > pd.Timedelta(days=t)).sum()}")

    # replace missing description with the description of other invoices if StockCode is the same:
    df.sort_values(by='StockCode', inplace=True)
    df['Description'].fillna(method='ffill', inplace=True)
    df.to_csv(Path('../data/online_sales_dataset_description_imputed.csv'), index=False)













