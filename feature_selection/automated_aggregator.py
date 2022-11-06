# automated version of aggregate_fs_datasets.py:
from pathlib import Path
from typing import List

import pandas as pd

# Global variables:
threshold: float = 5 * 10 ** -3
file_name: str = 'feature_importance_all_fe_features.csv'


# Functions:
def get_features_above_threshold(file_path: Path, thresh: float, dataframe: pd.DataFrame) \
        -> pd.DataFrame:
    """
    Get the features from the feature importance report above a certain threshold.
    @param file_path: the path to the csv file.
    @param thresh: the threshold.
    @param dataframe: the dataframe to get the features from.
    :return: the list of features above the threshold.
    """
    # read the csv file, the first row is the header:
    df_importance = pd.read_csv(file_path, index_col=0, header=0)

    # get the feature names with a value above the threshold:
    features_above_threshold = df_importance[df_importance['importance'] > thresh].index.tolist()

    # keep only those features and the CustomerId in the dataset:
    dataframe = dataframe[['CustomerId'] + features_above_threshold]

    return dataframe


if __name__ == '__main__':
    # read the dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_for_fs.csv'))

    # get the features above the threshold:
    df = get_features_above_threshold(Path('..', 'data', file_name), threshold, df)
    # save the dataset:
    df.to_csv(Path('..', 'data', 'online_sales_dataset_for_dr_automatic_thresholding.csv'))