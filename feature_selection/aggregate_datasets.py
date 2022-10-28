# Auxiliary library to combine the results from the wrapper methods
# based on feature commonality and performance with the tuned XGB model
# with embedded methods based on feature importance.

# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Driver:
if __name__ == '__main__':
    # import the dataset from recursive feature elimination:
    rfe = pd.read_csv(Path('..', 'data', 'online_sales_dataset_fs_rfe.csv'), index_col=0)
    # import the dataset from forwards selection:
    fs = pd.read_csv(Path('..', 'data', 'online_sales_dataset_fs_forward_selection.csv'), index_col=0)
    # import the dataset from backwards selection:
    bs = pd.read_csv(Path('..', 'data', 'online_sales_dataset_fs_backward_selection.csv'), index_col=0)
    # import the dataset with shared features from backwards and forwards selection:
    fs_bs = pd.read_csv(Path('..', 'data', 'online_sales_dataset_fs_forward_and_backward_selection.csv'), index_col=0)

    # get the features in all the datasets:
    rfe_features = set(rfe.columns)
    fs_bs_features = set(fs_bs.columns)
    intersection = rfe_features.intersection(fs_bs_features)
    print(f"The number of features in the intersection of RFE and FS+BS is: {len(intersection)}")
