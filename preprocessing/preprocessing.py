# Libraries:
import pandas as pd
from pathlib import Path
from data_imputation import customer_remover, missing_description_imputer, \
    stock_code_remover, stock_code_cleaner, parallelized_cancelling_order_imputer, \
    price_imputer
from data_loading import load_and_save_data

# Driver code:
if __name__ == '__main__':
    # Data Loading:
    # --------------------------------------------------------------
    df = load_and_save_data()

    # Data Cleaning:
    # --------------------------------------------------------------
    # remove duplicates:
    df.drop_duplicates()

    # remove missing costumer id's:
    df = customer_remover(df)

    # remove the stock codes of postage, sample, test and other non product stock items:
    df = stock_code_remover(df)

    # impute the missing values in the description column:
    df = missing_description_imputer(df)

    # sort the dataset by customer ID and date:
    df.sort_values(by=['Customer ID', 'InvoiceDate'], inplace=True)

    # rename Customer ID to CustomerId for consistency:
    df.rename(columns={'Customer ID': 'CustomerId'}, inplace=True)

    # Cast the CustomerId column to int:
    df['CustomerId'] = df['CustomerId'].astype(int)

    # impute the cancelling orders:
    df = parallelized_cancelling_order_imputer(df)

    # clean the stock codes to remove product variants, done after recovering the cancelling orders:
    df = stock_code_cleaner(df)

    # impute the values in the price column:
    df = price_imputer(df)

    # Create an enum for the countries:
    df['Country'] = df['Country'].astype('category')
    df['Country'] = df['Country'].cat.codes

    # Summary:
    # --------------------------------------------------------------
    # check that no missing values are left:
    print("Missing values:")
    print(f"{df.isna().sum()}")

    # save the dataset:
    df.to_csv(Path("..", "data", "online_sales_dataset_for_fe.csv"), index=False)

    # check the size of the dataset:
    print(f"Number of rows in the dataset: {df.shape[0]}")
    print(f"Number of unique customers: {df['CustomerId'].unique().size}")
    print(f"Number of unique invoices: {df['Invoice'].unique().size}")
    print(f"Number of unique products: {df['StockCode'].unique().size}")
    print(f"Number of unique descriptions: {df['Description'].unique().size}")
    print(f"Number of unique countries: {df['Country'].unique().size}")