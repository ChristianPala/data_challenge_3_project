# Libraries:
import pandas as pd
from pathlib import Path
from data_imputation import cancelling_order_remover, customer_remover, missing_description_imputer


# Driver code:
if __name__ == '__main__':
    # Load data
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset.csv'))

    # Data Cleaning:
    # --------------------------------------------------------------
    # remove duplicates:
    df.drop_duplicates()

    # remove missing costumer ids:
    df = customer_remover(df)

    # remove cancellation orders:
    df = cancelling_order_remover(df)

    # impute the missing values in the description column:
    df = missing_description_imputer(df)

    # fix the date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')

    # sort the dataset by customer ID and date:
    df.sort_values(by=['Customer ID', 'InvoiceDate'], inplace=True)

    # delete all bad debt, carriage, manual, postage, sample and test stock ids:
    df = df[~df['StockCode'].str.match('BAD|C2|DOT|M|POST|S|TEST', na=False)]

    # Now that we have only integers in the invoice column, we can convert it to int:
    df['Invoice'] = df['Invoice'].astype(int)

    # Replace the comma with a dot in the price column:
    df['Price'] = df['Price'].str.replace(',', '.')
    # Cast the price column to float:
    df['Price'] = df['Price'].astype(float)

    # Create an enum for the countries:
    df['Country'] = df['Country'].astype('category')
    df['Country'] = df['Country'].cat.codes

    # check the size of the dataset:
    print(f"Number of rows in the dataset: {df.shape[0]}")
    print(f"Number of unique customers: {df['Customer ID'].unique().size}")
    print(f"Number of unique invoices: {df['Invoice'].unique().size}")
    print(f"Number of unique products: {df['StockCode'].unique().size}")
    print(f"Number of unique descriptions: {df['Description'].unique().size}")

    # rename Customer ID to CustomerId:
    df.rename(columns={'Customer ID': 'CustomerId'}, inplace=True)

    # check that no missing values are left:
    print("Missing values:")
    print(f"{df.isna().sum()}")

    # save the dataset:
    df.to_csv(Path("..", "data", "online_sales_dataset_cleaned_v2.csv"), index=False)
