# Libraries:
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


if __name__ == '__main__':

    # import the cleaned dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_cleaned.csv'))

    # set some thresholds:
    thresholds = np.arange(0, 730, 30)

    # add a year for significance to the ndarray:
    thresholds = np.append(thresholds, 365)

    # cast LastPurchase to datetime:
    df['LastPurchase'] = df.groupby('CustomerId')['InvoiceDate'].transform('max')
    df['LastPurchase'] = pd.to_datetime(df['LastPurchase'])

    # check how many customers would have churned for each threshold:
    for threshold in thresholds:
        churned = df[df['LastPurchase'] < (pd.to_datetime('2011-12-31') - pd.Timedelta(days=threshold))]['CustomerId']\
            .nunique()
        print(f'For a hard threshold of {threshold} days from the end of the dataset, '
              f'{churned} customers would have churned.')

    # Considering as churners all customers that have not made a purchase in the last 365 days,
    # seems like a reasonable threshold. We will use this as a hard threshold for the churned customers.
