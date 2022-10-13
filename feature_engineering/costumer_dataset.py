# Libraries:
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from preprocessing.data_imputation import cancelling_order_remover


if __name__ == '__main__':
    # import the cleaned dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_cleaned.csv'))

    # create the aggregated costumer dataset for quantities, rename to TotalQuantity:
    df_agg = df.groupby('CustomerId').agg({'Quantity': 'sum'}).rename(columns={'Quantity': 'TotalQuantity'})
    # if the total quantity is smaller or equal to 0, delete the customer:
    df_agg = df_agg[df_aggt['TotalQuantity'] > 0]

    # create a total spent column, price * quantity:
    df['TotalSpent'] = df['Price'] * df['Quantity']
    df_agg['TotalSpent'] = df.groupby('CustomerId').agg({'TotalSpent': 'sum'})
    # round the TotalSpent to 2 decimals:
    df_agg['TotalSpent'] = df_agg['TotalSpent'].round(2)
    # if the total spent is smaller or equal to 0, delete the customer:
    df_agg = df_agg[df_agg['TotalSpent'] > 0]

    # now that we have corrected the cancellation orders on a customer ID level, we can aggregate on the
    # rest of the dataset after dropping those rows:
    df = cancelling_order_remover(df)

    # aggregate the other columns:
    df_agg[['NumberOfPurchases', 'Description', 'LastPurchase', 'Country']] = \
        df.groupby('CustomerId').agg({'Invoice': 'count', 'Description': lambda x: ' '.join(x), 'InvoiceDate': 'max',
                                      'Country': lambda x: x.value_counts().index[0]})

    # Use the above for our first definition of churn, costumers that have not purchased in the last @timeframe months:
    timeframe = 365
    df_agg['LastPurchase'] = pd.to_datetime(df_agg['LastPurchase'])
    df_agg['CustomerChurned'] = df_agg['LastPurchase'] < datetime(2011, 12, 31) - timedelta(days=timeframe)

    # Delete the variable last purchase as it is a proxy for the target variable:
    df_agg.drop('LastPurchase', axis=1, inplace=True)

    # check how many churned costumers we have:
    print(f'Number of churned costumers: {df_agg["CustomerChurned"].sum()}')

    # check the number of customers:
    print(f'Number of customers: {df_agg.shape[0]}')

    # save the dataset:
    df_agg.to_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # save the churned costumers in a separate dataset:
    df_agg[df_agg['CustomerChurned']]['CustomerChurned'].to_csv(Path('..', 'data',
                                                                     'online_sales_dataset_agg_churned.csv'))
