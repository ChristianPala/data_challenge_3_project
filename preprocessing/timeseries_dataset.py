# Libraries
import pandas as pd
from pathlib import Path
from data_imputation import cancelling_order_remover, stock_code_cleaner


if __name__ == '__main__':
    # import the cleaned dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_cleaned.csv'))
    # remove the description column:
    df.drop('Description', axis=1, inplace=True)

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).astype(int) / 10**9

    df = cancelling_order_remover(df)

    df = stock_code_cleaner(df)

    print(df.isna().sum())

    df.to_csv(Path('..', 'data', 'online_sales_dataset_timeseries.csv'), index=False)
