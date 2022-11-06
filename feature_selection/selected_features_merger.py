# Libraries
# Data manipulation
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    # load data
    forward_sel = pd.read_csv(Path('data', 'online_sales_dataset_fs_forward_selection.csv'))
    backward_sel = pd.read_csv(Path('data', 'online_sales_dataset_fs_backward_selection.csv'))
    recursive_sel = pd.read_csv(Path('data', 'online_sales_dataset_fs_rfe.csv'))

    if 'CustomerId' not in backward_sel.columns:
        backward_sel['CustomerId'] = forward_sel['CustomerId']

    # merge the datasets keeping all the features:
    merged = pd.merge(forward_sel, backward_sel, on='CustomerId', how='outer')
    merged = pd.merge(merged, recursive_sel, on='CustomerId', how='outer')
    # remove duplicate features:
    merged = merged.loc[:, ~merged.columns.duplicated()]
    # save the dataset:
    merged.to_csv(Path('data', 'online_sales_dataset_fs_wrapper_union.csv'), index=False)

    # get the column names:
    forward_sel_cols = forward_sel.columns
    backward_sel_cols = backward_sel.columns
    recursive_sel_cols = recursive_sel.columns
    # get the common features:
    common_features = np.intersect1d(forward_sel_cols, backward_sel_cols)
    common_features = np.intersect1d(common_features, recursive_sel_cols)
    # keep only the common features in the merged dataset:
    intersected = merged[common_features]
    # save the dataset:
    intersected.to_csv(Path('data', 'online_sales_dataset_fs_wrapper_intersection.csv'), index=False)

    # print the shapes of the datasets:
    print(f'Forward selection shape: {forward_sel.shape}')
    print(f'Backward selection shape: {backward_sel.shape}')
    print(f'Recursive feature elimination shape: {recursive_sel.shape}')
    print(f'Merged dataset shape: {merged.shape}')
    print(f'Intersected dataset shape: {intersected.shape}')


if __name__ == '__main__':
    main()
