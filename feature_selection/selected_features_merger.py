# Libraries
# Data manipulation
import numpy as np
import pandas as pd
from pathlib import Path


def union(df_1, df_2, df_3):
    """

    @param df_1: the dataframe that have to be unified
    @param df_2: the dataframe that have to be unified
    @param df_3: the dataframe that have to be unified
    @return: the unified dataframe
    """
    uni = pd.merge(df_1, df_2, how='outer')
    return pd.merge(uni, df_3, how='outer')


def intersect(df_1, df_2, df_3):
    """

    @param df_1: the 1st dataframe that have to be intersected
    @param df_2: the 2nd dataframe that have to be intersected
    @param df_3: the 3rd dataframe that have to be intersected
    @return: the intersected dataframe
    """
    inter = pd.merge(df_1, df_2, how='inner')
    return pd.merge(inter, df_3, how='inner')


if __name__ == '__main__':
    forward_sel = pd.read_csv(Path('..', 'data', 'online_sales_dataset_fs_forward_selection.csv'))
    backward_sel = pd.read_csv(Path('..', 'data', 'online_sales_dataset_fs_backward_selection.csv'))
    recursive_sel = pd.read_csv(Path('..', 'data', 'online_sales_dataset_fs_rfe.csv'))

    unified = union(forward_sel, backward_sel, recursive_sel)
    intersected = intersect(forward_sel, backward_sel, recursive_sel)

    print(unified.shape)
    print(intersected.shape)

    unified.to_csv(Path('..', 'data', 'online_sales_dataset_fs_bs_rfe_union.csv'), index=False)
    intersected.to_csv(Path('..', 'data', 'online_sales_dataset_fs_bs_rfe_intersect.csv'), index=False)


