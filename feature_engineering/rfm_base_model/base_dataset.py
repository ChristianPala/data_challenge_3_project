# Aggregate the dataset by customer for the base model, use the RFM strategy to select features.
# Libraries:
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def main():
    # import the cleaned dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))

    # create the aggregated costumer dataset for quantities, rename to TotalQuantity:
    df_agg = df.groupby('CustomerId').agg({'Quantity': 'sum'}).rename(columns={'Quantity': 'TotalQuantity'})
    # if the total quantity is smaller or equal to 0, delete the customer:
    df_agg = df_agg[df_agg['TotalQuantity'] > 0]

    # create a total spent column, price * quantity:
    df['TotalSpent'] = df['Price'] * df['Quantity']
    df_agg['TotalSpent'] = df.groupby('CustomerId').agg({'TotalSpent': 'sum'})
    # round the TotalSpent to 2 decimals:
    df_agg['TotalSpent'] = df_agg['TotalSpent'].round(2)
    # if the total spent is smaller or equal to 0, delete the customer:
    df_agg = df_agg[df_agg['TotalSpent'] > 0]

    # aggregate the other columns:
    df_agg[['NumberOfPurchases', 'NumberOfProducts', 'Description', 'LastPurchase', 'Country']] = \
        df.groupby('CustomerId').agg({'Invoice': 'count', 'StockCode': 'nunique',
                                      'Description': lambda x: ' '.join(x), 'InvoiceDate': 'max',
                                      'Country': lambda x: x.value_counts().index[0]})

    # Use the above for our first definition of churn, costumers that have not purchased in the last @timeframe months:
    timeframe = 365
    reference_date = datetime(2011, 12, 10)
    churn_date = reference_date - timedelta(days=timeframe)
    df_agg['LastPurchase'] = pd.to_datetime(df_agg['LastPurchase'])
    df_agg['CustomerChurned'] = df_agg['LastPurchase'] < churn_date

    # Delete the variable last purchase as it is a proxy for the target variable:
    df_agg.drop('LastPurchase', axis=1, inplace=True)

    # check how many churned costumers we have:
    print(f'Number of churned costumers: {df_agg["CustomerChurned"].sum()}')

    # check the number of customers:
    print(f'Number of customers: {df_agg.shape[0]}')

    # place the target variable at the end of the dataset:
    df_agg = df_agg[['NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'NumberOfProducts',
                     'Description', 'Country', 'CustomerChurned']]

    # save the dataset:
    df_agg.to_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    # save the churned costumers in a separate dataset:
    df_agg[df_agg['CustomerChurned']]['CustomerChurned'].to_csv(Path('..', '..', 'data',
                                                                     'online_sales_dataset_agg_churned.csv'))

    # check for missing values:
    print(df_agg.isnull().sum())

    # consistency check between the customer dataset and the dataset for feature engineering:
    fe_customers = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))['CustomerId'].unique()

    df_fe_customers = pd.DataFrame(fe_customers, columns=['CustomerId'])

    if df_agg.shape[0] != df_fe_customers.shape[0]:
        # print the difference:
        print(df_fe_customers[~df_fe_customers['CustomerId'].isin(df_agg.index)])
    else:
        print("Consistency check passed, customer dataset and dataset for feature engineering have the same number of "
              "customers.")


# Driver:
if __name__ == '__main__':
    main()
