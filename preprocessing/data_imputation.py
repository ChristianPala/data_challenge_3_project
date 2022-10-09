# Libraries:
import pandas as pd


# Functions:
def cancelling_order_remover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes cancelled orders from the dataset.
    @param df: dataframe with the cancelled orders.
    :return: dataframe with the cancelled orders removed.
    """
    # remove the cancelled orders:
    df.drop(df[df['Invoice'].str.startswith('C', na=False)].index, inplace=True)
    # na=false does not select empty invoices, not necessary here but good practice.
    return df


def missing_description_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes the missing descriptions in the dataset, which can be recovered with the stock code.
    @param df: dataframe with the missing descriptions.
    :return: dataframe with the missing descriptions imputed.
    """
    # impute the missing values in the description column:
    df['Description'] = df['Description'].fillna(df['StockCode'])
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
