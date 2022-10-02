# Libraries:
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


if __name__ == '__main__':

    # import the cleaned dataset:
    df = pd.read_csv(Path('../data/online_sales_dataset_cleaned.csv'))

    # set some thresholds:
    thresholds = [31, 62, 93, 186, 365]

    # cast last_purchase to datetime:
    df['last_purchase'] = pd.to_datetime(df['last_purchase'], format='%Y-%m-%d %H:%M')

    # check how many customers would have churned for each threshold:
    for threshold in thresholds:
        churned = df[df['last_purchase'] < (pd.to_datetime('2011-12-31') - pd.Timedelta(days=threshold))]['Customer ID'].nunique()
        print(f'For a hard threshold of {threshold} days from the end of the dataset, '
              f'{churned} customers would have churned.')
    