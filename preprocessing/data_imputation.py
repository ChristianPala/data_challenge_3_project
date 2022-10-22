# Libraries:
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# globals:
missing_counter = 0


# Functions:

def sort_by_date_given_reference(df: pd.DataFrame, reference_date) -> pd.DataFrame:
    """
    Auxiliary function for cancelling_order_imputer() to sort the dataframe by the date given a reference date.
    @param df: the dataframe to be sorted.
    @param reference_date: the reference date to sort the dataframe by.
    :return: the sorted dataframe.
    """

    df['DateDiff'] = abs(df['InvoiceDate'] - reference_date)
    df.sort_values(by=['DateDiff'], inplace=True)
    df.drop(columns=['DateDiff'], inplace=True)
    return df


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
    # delete Manual descriptions:
    df = df[~df['Description'].str.startswith('Manual')]
    # impute the missing values in the description column:
    df['Description'] = df['Description'].fillna(df['StockCode'])
    return df


def stock_code_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes trailing letters from the stock code, used to perform the timeseries analysis
    @param df: dataframe already preprocessed with stock_code_remover() function
    :return: cleaned dataframe, containing integer-only stock codes
    """
    # if a stock code has non-numeric characters, remove them:
    df['StockCode'] = df['StockCode'].str.replace('[^0-9]', '', regex=True)

    # if a stock code is empty or has white spaces, remove it:
    df['StockCode'] = df['StockCode'].str.strip()
    df.drop(df[df['StockCode'] == ''].index, inplace=True)

    # cast the stock code to integer:
    df['StockCode'] = df['StockCode'].astype(int)

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
    @param df: dataframe with the stock codes.
    :return: dataframe with the irrelevant stock codes removed.
    """
    # delete all bad debt, carriage, manual, postage, sample and test stock ids:
    df = df[~df['StockCode'].str.startswith('B|C2|D|DOT|M|POST|S|TEST')]
    return df


def cancelling_order_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    worker function for parallelized_cancelling_order_cleaner()
    @param df: dataframe with the cancelling orders.
    :return: dataframe with the cancelling orders removed."""

    global missing_counter

    for row in tqdm(df.itertuples()):
        # if the invoice starts with C, it is a cancelling order:
        if row.Invoice.startswith('C'):
            # find the corresponding positive order:
            original_orders = df[(df['CustomerId'] == row.CustomerId)
                                 # remove the hours and minutes from the date
                                 # assume the original order was between one day before and one day after the
                                 # cancelling order
                                 & ((df['InvoiceDate'].dt.date == row.InvoiceDate.date() + pd.Timedelta(days=1))
                                    | (df['InvoiceDate'].dt.date == row.InvoiceDate.date())
                                    | (df['InvoiceDate'].dt.date == row.InvoiceDate.date() - pd.Timedelta(days=1)))
                                 & (df['StockCode'] == row.StockCode)].copy()
            # check if the positive order exists:
            if original_orders.empty:
                # if no positive order exists, remove the cancelling order:
                missing_counter += 1
                df.drop(row.Index, inplace=True)
                continue

            # sort the original orders by the cancellation date:
            original_orders = sort_by_date_given_reference(original_orders, row.InvoiceDate)

            for order in original_orders.itertuples():
                # if the positive order exists, and the quantity is greater than the cancelling order,
                # subtract the quantity of the cancelling order from the positive order
                # and remove the cancelling order:
                if order.Quantity == -row.Quantity:
                    df.drop(order.Index, inplace=True)
                    df.drop(row.Index, inplace=True)
                    break
                if order.Quantity > -row.Quantity:
                    df.at[order.Index, 'Quantity'] += row.Quantity
                    df.drop(row.Index, inplace=True)
                    break
            else:
                df.drop(row.Index, inplace=True)
                missing_counter += 1

    print(f"Missing positive orders: {missing_counter}")

    return df


def parallelized_cancelling_order_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrects all the cancelling orders from the dataset in parallel.
    @param df: dataframe with the cancelling orders.
    :return: dataframe with the cancelling orders removed."""

    # format the invoice date as a datetime object, if it is not already:
    if not isinstance(df['InvoiceDate'].iloc[0], pd.Timestamp):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')

    # rename Customer ID to CustomerId if it is not already:
    if 'Customer ID' in df.columns:
        df.rename(columns={'Customer ID': 'CustomerId'}, inplace=True)

    # order the dataframe by customer id, invoice date and stock code:
    df.sort_values(by=['CustomerId', 'InvoiceDate', 'StockCode'], inplace=True)

    # number of cancelling orders:
    cancelling_orders = len(df[df['Invoice'].str.startswith('C')])

    # get the partitions size for the local machine:
    nr_partitions = mp.cpu_count() - 1

    # split the dataframe into required number of partitions
    partitions = np.array_split(df, nr_partitions)
    # note in a very unlucky case we might split a customer's orders into different partitions
    # and the cancelling order imputer will not work properly, we checked the chunks and
    # for our local machine it was not the case.

    # create a pool of workers with the local machine's number of partitions:
    pool = mp.Pool(nr_partitions)

    # apply the cancelling_order_cleaner function to each partition:
    chunks = pool.map(cancelling_order_imputer, partitions)

    # merge the chunks:
    df = pd.concat(chunks)

    print(f"Missing positive orders: {missing_counter}")
    print("Recovered cancelling orders: ", cancelling_orders - missing_counter)

    return df


def price_imputer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes prices incorrectly set to 0.
    @param df: dataframe to impute
    :return: dataframe with the prices imputed.
    """

    # Replace the comma with a dot in the price column:
    df['Price'] = df['Price'].astype(str).str.replace(',', '.')
    # Cast the price column to float:
    df['Price'] = df['Price'].astype(float)
    # Round the price column to 2 decimals:
    df['Price'] = df['Price'].round(2)

    # get rows with price 0 or lower:
    zero_price_rows = df[df['Price'] <= 0.00]
    # check if there are rows with the same stock code, but different prices:
    for row in zero_price_rows.itertuples():
        # get all the rows with the same stock code:
        same_stock_code_rows = df[df['StockCode'] == row.StockCode]
        # get the unique prices of the rows with the same stock code:
        unique_prices = same_stock_code_rows['Price'].unique()
        # if there is only one unique price, impute the price of the row with the unique price:
        if len(unique_prices) == 1:
            df.at[row.Index, 'Price'] = unique_prices[0]
        # if there are multiple unique prices, impute the price of the row with the median price:
        else:
            df.at[row.Index, 'Price'] = same_stock_code_rows['Price'].median()

    # if the price is still 0 or lower, delete the row:
    df = df[df['Price'] > 0.00]

    return df