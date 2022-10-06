# Libraries:
import pandas as pd
from pathlib import Path
from imputer import cancelling_order_imputer, missing_descriptions_imputer


# Driver code:
if __name__ == '__main__':
    # data file_paths
    f_09 = Path('..', 'data', 'online_sales_2009_2010_dataset.csv')
    f_10 = Path('..', 'data', 'online_sales_2010_2011_dataset.csv')
    # Load data
    df_09 = pd.read_csv(f_09, sep=';')
    df_10 = pd.read_csv(f_10, sep=';')
    # merge the two datasets for preprocessing:
    pd.concat([df_09, df_10], axis=0).to_csv(Path('..', 'data', 'online_sales_dataset.csv'), index=False)
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset.csv'))

    # remove duplicates:
    df.drop_duplicates()

    # drop all rows with missing costumer id:
    # From Christian Berchtold's analysis, and discussing with professor Mitrovic, we cannot recover them.
    df.dropna(subset=['Customer ID'], inplace=True)

    # impute the cancelled orders:
    df = cancelling_order_imputer(df)

    # impute the missing values in the description column:
    df = missing_descriptions_imputer(df)

    # fix the date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')

    # sort the dataset by customer ID and date:
    df.sort_values(by=['Customer ID', 'InvoiceDate'], inplace=True)

    # delete all bad debt, carriage, manual, postage, sample and test stock ids:
    df = df[~df['StockCode'].str.contains('B|C2|DOT|M|POST|S|TEST', case=False)]

    # Now that we have only integers in the invoice column, we can convert it to int:
    df['Invoice'] = df['Invoice'].astype(int)

    # maybe this is not needed.
    # check if the last character of the stock code is a letter, if so remove the letter since we noticed they are just
    # different versions of the same product:
    # df['StockCode'] = df['StockCode'].str.replace(r'[a-zA-Z]+$', '', regex=True)

    # drop empty stock codes:
    df = df[df['StockCode'] != '']

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

    # save the dataset:
    df.to_csv(Path("..", "data", "online_sales_dataset_cleaned.csv"), index=False)
