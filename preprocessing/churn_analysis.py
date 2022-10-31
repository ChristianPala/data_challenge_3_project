# Libraries:
# Data manipulation:
import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == '__main__':

    # import the cleaned dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_for_fe.csv'))

    # set some thresholds:
    thresholds = np.arange(0, 730, 30)

    # cast LastPurchase to datetime:
    df['LastPurchase'] = df.groupby('CustomerId')['InvoiceDate'].transform('max')
    df['LastPurchase'] = pd.to_datetime(df['LastPurchase'])

    # check how many customers would have churned for each threshold:
    for threshold in thresholds:
        churned = df[df['LastPurchase'] < (pd.to_datetime('2011-12-31') - pd.Timedelta(days=threshold))]['CustomerId'] \
            .nunique()
        print(f'For a hard threshold of {threshold} days from the end of the 2011, '
              f'{churned} customers would have churned.')

    # get the last purchase date in the dataset:
    last_purchase = df['LastPurchase'].max()
    # subtract 365 days from the last purchase date:
    last_purchase_365 = last_purchase - pd.Timedelta(days=365)

    # get the number of customers that have not purchased in the last 365 days:
    churned_365 = df[df['LastPurchase'] < last_purchase_365]['CustomerId'].nunique()
    print(f'For a hard threshold of 365 days from the last purchase in the dataset, '
          f'{churned_365} customers would have churned.')
    # Considering as churners all customers that have not made a purchase in the last 365 days from the last
    # available purchase date in the dataset seems like a reasonable threshold.
    # We will use this as a hard threshold for the churned customers.
