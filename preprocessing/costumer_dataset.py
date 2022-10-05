# Libraries:
import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    # import the cleaned dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_cleaned.csv'))

    # create the aggregated costumer dataset:
    df_agg = df.groupby('CustomerId').agg({'Invoice': 'count', 'Quantity': 'sum', 'Price': 'sum',
                                           'Description': ' '.join, 'Country': lambda x: x.value_counts().index[0]})
    df_agg.rename(columns={'Invoice': 'NumberOfPurchases', 'Quantity': 'TotalQuantity', 'Price': 'TotalSpent'},
                  inplace=True)

    # create a new column with the last purchase date for each costumer and convert it to datetime:
    df_agg['LastPurchase'] = df.groupby('CustomerId')['InvoiceDate'].max()
    df_agg['LastPurchase'] = pd.to_datetime(df_agg['LastPurchase'], format='%Y-%m-%d %H:%M')

    # Use the above for our first definition of churn, costumers that have not purchased in the last 12 months:
    df_agg['CustomerChurned'] = df_agg['LastPurchase'] < pd.to_datetime('2010-12-01')

    # Delete the variable last purchase as it is a proxy for the target variable:
    df_agg.drop('LastPurchase', axis=1, inplace=True)

    # check the number of churned customers:
    print(f'Number of churned customers: {df_agg["CustomerChurned"].sum()}')

    # check the number of customers:
    print(f'Number of customers: {df_agg.shape[0]}')

    # save the dataset:
    df_agg.to_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))
