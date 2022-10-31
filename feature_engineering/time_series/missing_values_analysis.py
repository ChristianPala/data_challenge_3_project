# Libraries:
import pandas as pd
from pathlib import Path

# Driver:
if __name__ == '__main__':
    # load the dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_tsfel.csv'), index_col=0)

    # check how many features we have:
    print(f"Number of features: {df.shape[1]}")

    # drop columns with more than 70% missing values:
    threshold = len(df) * .70
    df_dropped = df.dropna(thresh=threshold, axis=1).copy()

    # check how many columns were dropped:
    print(f"Number of features dropped: {df.shape[1] - df_dropped.shape[1]}")

    # check how many columns still have missing values:
    missing_values_columns = [i for i in df_dropped.columns if df_dropped[i].isnull().any()]
    print(f"Number of features with missing values: {len(missing_values_columns)}")

    # for each column with missing values, fill the missing values with the median of the column:
    for col in df_dropped.columns:
        if df_dropped[col].isnull().sum() > 0:
            df_dropped[col].fillna(df_dropped[col].median(), inplace=True)

    # check the number of missing values:
    print(f"Number of missing values: {df_dropped.isnull().sum().sum()}")

    # save the dataset:
    df_dropped.to_csv(Path('..', '..', 'data', 'online_sales_dataset_tsfel_for_fs.csv'))