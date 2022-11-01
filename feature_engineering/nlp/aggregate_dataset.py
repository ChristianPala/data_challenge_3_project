# Auxiliary library to collect the nlp features:

import pandas as pd
from pathlib import Path

# Driver:
if __name__ == '__main__':
    # load the nlp features:
    df_train = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_train.csv'), index_col=0)
    df_test = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_test.csv'), index_col=0)

    # if there are missing values raise an exception:
    if df_train.isnull().sum().sum() > 0:
        raise Exception('There are missing values in the train dataset.')
    if df_test.isnull().sum().sum() > 0:
        raise Exception('There are missing values in the test dataset.')

    # merge the train and test datasets:
    df = pd.concat([df_train, df_test], axis=0)

    # exclude all non cluster columns:
    df_final = df[[col for col in df.columns if 'Cd_' in col]].copy()

    # add the DescriptionLength column:
    df_final['DescriptionLength'] = df['DescriptionLength']

    # save the dataset:
    df_final.to_csv(Path('..', '..', 'data', 'online_sales_dataset_nlp_for_fs.csv'))