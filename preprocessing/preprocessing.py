# Libraries:
import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == '__main__':

    # very rough cleaning:
    # data file_paths
    f_09 = Path('..', 'data', 'online_sales_2009_2010_dataset.csv')
    f_10 = Path('..', 'data', 'online_sales_2010_2011_dataset.csv')
    # Load data
    df_09 = pd.read_csv(f_09, sep=';')
    df_10 = pd.read_csv(f_10, sep=';')
    # merge the two datasets for preprocessing:
    pd.concat([df_09, df_10], axis=0).to_csv(Path('..', 'data', 'online_sales_dataset.csv'), index=False)
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset.csv'))

    # recovering most of the missing descriptions:
    # get the rows with missing descriptions but with a matching stock code with a non-missing description:
    missing_descriptions_with_matching_stock_code_in_db = df[df['Description'].isna()
                                                             & df['StockCode'].isin(df[df['Description']
                                                                                    .notna()]['StockCode'])].index
    for i in missing_descriptions_with_matching_stock_code_in_db:
        df.loc[i, 'Description'] = df[df['StockCode'] == df.loc[i, 'StockCode']]['Description'].values[0]
    # Drop the remaining missing descriptions which cannot be recovered:
    df.dropna(subset=['Description'], inplace=True)

    # remove duplicates:
    df.drop_duplicates()

    # drop all rows with missing costumer id:
    # From Chri B.'s analysis we cannot recover them.
    df.dropna(subset=['Customer ID'], inplace=True)

    # fix the date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')
    # sort the dataset by customer ID and date:
    df.sort_values(by=['Customer ID', 'InvoiceDate'], inplace=True)

    # delete all bad debt, carriage, manual, postage, sample and test stock ids:
    df = df[~df['StockCode'].str.contains('B|C2|DOT|M|POST|S|TEST', case=False)]

    # create a list of invoices starting with C, removing the C from the invoice number:
    cancelled_invoices = df[df['Invoice'].str.startswith('C')]['Invoice'].str[1:].tolist()

    # look for partial cancellations

    # delete all rows with invoices matching cancelled_invoices:
    df = df[~df['Invoice'].isin(cancelled_invoices)]

    # delete all rows with invoices starting with C:
    df = df[~df['Invoice'].str.startswith('C')]

    # if you see a product in the cancelled with product P with price Y, you can have another transaction with anothe

    # Now that we have only integers in the invoice column, we can convert it to int:
    df['Invoice'] = df['Invoice'].astype(int)

    # check if the last character of the stock code is a letter, if so remove the letter since we noticed they are just
    # different versions of the same product:
    df['StockCode'] = df['StockCode'].str.replace(r'[a-zA-Z]+$', '', regex=True)

    # drop empty stock codes:
    df = df[df['StockCode'] != '']

    # We still have some literal characters (like the gift vouchers, so we will treat stock codes as strings):
    df['StockCode'] = df['StockCode'].astype(str)

    # Descriptions should be strings, we will try to feature engineer them later:
    df['Description'] = df['Description'].astype(str)

    # Replace the comma with a dot in the price column:
    df['Price'] = df['Price'].str.replace(',', '.')
    # Cast the price column to float:
    df['Price'] = df['Price'].astype(float)

    # Create an enum for the countries:
    df['Country'] = df['Country'].astype('category')
    df['Country'] = df['Country'].cat.codes

    # create a new column with the last purchase date for each costumer and convert it to datetime:
    df['LastPurchase'] = df.groupby('Customer ID')['InvoiceDate'].max()
    df['LastPurchase'] = pd.to_datetime(df['LastPurchase'], format='%Y-%m-%d %H:%M')

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
