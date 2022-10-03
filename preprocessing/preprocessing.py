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

    # try to recover the missing descriptions:

    # create indices for the missing descriptions:
    # missing_descriptions_pure = df[df['Description'].isna()].index
    # print(len(missing_descriptions_pure))
    # filter the missing descriptions, keep only those which have a matching stock code with a non empty description
    # in the dataset:
    # missing_descriptions_filtered = df[df['Description'].isna() & df['StockCode'].isin(df[df['Description']
    #                                                                                   .notna()]['StockCode'])].index
    # print(len(missing_descriptions_filtered))

    # We can recover 4019 descriptions, so we'll do the matching even if it's slow:

    # Codice Chri B.
    # # if there are same stock codes with different descriptions, keep the most frequent one
    # sc = df['StockCode']
    # equal_stockCodes = set(sc[sc.duplicated()].tolist())  # set to keep just one number per occurrence
    # # get descriptions corresponding to each stock code
    # descriptions = {}
    # c = 0
    # for s in equal_stockCodes:
    #     descriptions[s] = str(df.loc[df['StockCode'] == s]['Description'].str.strip().mode().tolist())
    #     print(f'\r{round(c/len(equal_stockCodes)*100)}% completed', end='', flush=True)
    #     c += 1
    # print('')
    # descriptions = pd.DataFrame(descriptions, columns=['s', 'Description'])  # this is the dataframe containing every duplicate stock code and its corresponding most frequent description
    # # man i dont get it... how to substitute all the descriptions??

    for i in df[df['Description'].isna()].index:
        if df[df['StockCode'] == df.loc[i, 'StockCode']]['Description'].notna().sum() >= 1:
            # take the first non-empty description and replace the missing one, we noticed articles with the same
            # stock code have the same basic description with small variations:
            df.loc[i, 'Description'] = df[df['StockCode'] == df.loc[i, 'StockCode']]['Description'].values[0]
        else:
            df.drop(i, axis=0, inplace=True)

    # we drop the rest
    df.dropna()
    df.drop_duplicates()

    # drop all rows with missing costumer id:
    # can we recover something?
    df = df[df['Customer ID'].notna()]

    # fix the date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')
    # sort the dataset by customer ID and date:
    df.sort_values(by=['Customer ID', 'InvoiceDate'], inplace=True)

    # delete all bad debt, carriage, manual, postage, sample and test stock ids:
    df = df[~df['StockCode'].str.contains('B|C2|DOT|M|POST|S|TEST', case=False)]

    # create a list of invoices starting with C, removing the C from the invoice number:
    cancelled_invoices = df[df['Invoice'].str.startswith('C')]['Invoice'].str[1:].tolist()

    # delete all rows with invoices matching cancelled_invoices:
    df = df[~df['Invoice'].isin(cancelled_invoices)]

    # delete all rows with invoices starting with C:
    df = df[~df['Invoice'].str.startswith('C')]

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
