# Libraries:
import pandas as pd
from pathlib import Path
from data_imputation import customer_remover, missing_description_imputer, stock_code_remover
from data_loading import load_and_save_data

# Driver code:
if __name__ == '__main__':
    # Load data
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

    # fix the date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')

    # Replace the comma with a dot in the price column:
    df['Price'] = df['Price'].str.replace(',', '.')
    # Cast the price column to float:
    df['Price'] = df['Price'].astype(float)

    # Create an enum for the countries:
    df['Country'] = df['Country'].astype('category')
    df['Country'] = df['Country'].cat.codes

    # sort the dataset by customer ID and date:
    df.sort_values(by=['Customer ID', 'InvoiceDate'], inplace=True)

    # rename Customer ID to CustomerId for consistency:
    df.rename(columns={'Customer ID': 'CustomerId'}, inplace=True)

    # check that no missing values are left:
    print("Missing values:")
    print(f"{df.isna().sum()}")

    # check the size of the dataset:
    print(f"Number of rows in the dataset: {df.shape[0]}")
    print(f"Number of unique customers: {df['CustomerId'].unique().size}")
    print(f"Number of unique invoices: {df['Invoice'].unique().size}")
    print(f"Number of unique products: {df['StockCode'].unique().size}")
    print(f"Number of unique descriptions: {df['Description'].unique().size}")
    print(f"Number of unique countries: {df['Country'].unique().size}")

    # save the dataset:
    df.to_csv(Path("..", "data", "online_sales_dataset_cleaned.csv"), index=False)
