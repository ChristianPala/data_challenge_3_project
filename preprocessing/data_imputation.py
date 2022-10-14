# Libraries:
import pandas as pd
import numpy as np


# Functions:
def missing_description_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes the missing descriptions in the dataset, which can be recovered with the stock code.
    @param df: dataframe with the missing descriptions.
    :return: dataframe with the missing descriptions imputed.
    """
    # remove "this is a test product" from the dataset:
    df.drop(df[df['Description'] == 'This is a test product.'].index, inplace=True)
    # remove "adjustment by" from the dataset:
    df.drop(df[df['Description'].str.startswith('Adjustment by', na=False)].index, inplace=True)
    # remove "POSTAGE" from the dataset:
    df.drop(df[df['Description'] == 'POSTAGE'].index, inplace=True)

    # impute the missing values in the description column:
    df['Description'] = df['Description'].fillna(df['StockCode'])
    return df


def stock_code_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes trailing letters from the stock code, used to perform the timeseries analysis
    :param df: dataframe already preprocessed with stock_code_remover() function
    :return: cleaned dataframe, containing integer-only stock codes
    """
    df['StockCode'].replace('[a-zA-Z]+', value='', regex=True, inplace=True)
    df['StockCode'].replace('', value=np.nan, inplace=True)

    df.dropna(subset=['StockCode'], inplace=True)  # dropping stray null
    return df


def customer_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the missing customer ids in the dataset. Left for possible future use, if we figure out how to impute them.
    @param df: dataframe with the missing customer ids.
    :return: dataframe with the missing customer ids removed
    """
    # delete the missing customer ids:
    df.drop(df[df['Customer ID'].isna()].index, inplace=True)
    return df


def stock_code_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the stock codes of postage, sample, test and other non product stock items.
    :param df: dataframe with the stock codes.
    :return: dataframe with the irrelevant stock codes removed.
    """
    # delete all bad debt, carriage, manual, postage, sample and test stock ids:
    df = df[~df['StockCode'].str.startswith('B|C2|D|DOT|M|POST|S|TEST')]
    # delete manual descriptions:
    df = df[~df['Description'].str.startswith('Manual')]
    return df


def cancelling_order_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    removes all cancelling orders from the dataset.
    :param df: dataframe with the cancelling orders.
    :return: dataframe with the cancelling orders removed.
    """
    # remove all the cancelling orders:
    df = df[~df['Invoice'].str.startswith('C')]
    return df
