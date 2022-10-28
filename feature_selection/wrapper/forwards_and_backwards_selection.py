# Libraries:
# Data manipulation:
import numpy as np
import pandas as pd
from pathlib import Path

# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from xgboost import XGBClassifier

# Time:
import time
import datetime


# Functions:
def feature_selection(estimator, x_tr, y_tr, direction: str = 'forward') -> np.array:
    """
    Function to perform feature selection on a given direction, default is forward.
    @param estimator: the model used to predict the target
    @param x_tr: train split of the X dataframe
    @param y_tr: train split of the target dataframe
    @param direction: either forward or backward
    @return: the mask of the selected features
    """
    sfs = SequentialFeatureSelector(estimator=estimator,
                                    direction=direction,
                                    n_features_to_select='auto',
                                    tol=0.0001,
                                    n_jobs=-1)
    print(f'> performing feature selection. Method: {direction}')
    sfs.fit(x_tr, y_tr)
    print(f'sfs_{direction} fitted')
    print(f'shape ({direction}):', sfs.transform(x_tr).shape)
    support = sfs.get_support()
    return support


if __name__ == '__main__':
    # import the dataset:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_mutual_information.csv'))
    df_fs = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fs.csv'))

    # make sure the dataset from mutual information preserves the order
    # of the dataset from feature selection
    try:
        # check that both df are sorted, so that customer ids match
        assert X.loc[0, 'CustomerId'] == df_fs.loc[0, 'CustomerId'], \
            'CustomerId are not matching (not sorted dataframes)'
    except AssertionError:
        # if not sorted, sort the dataframe and reset the index value
        X.sort_values(by='CustomerId', inplace=True)
        X.reset_index(inplace=True, drop=True)

    X.drop('CustomerId', axis=1, inplace=True)
    # import the label dataset:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # X = X[:50]  # slice for debugging
    # y = y[:50]  # slice for debugging

    feature_names = np.array(X.columns)
    # print(feature_names)

    # perform the train test split:
    X_train, X_test, y_train, y_test = \
        train_validation_test_split(X, y)

    # best = tuner(X_train, y_train, X_validation, y_validation, cross_validation=5)

    # define the model:
    model = XGBClassifier(objective="binary:logistic", random_state=42, n_jobs=-1)

    # perform feature selection:
    s = time.time()
    support_f = feature_selection(model, X_train, y_train, 'forward')
    e = time.time() - s
    print('time:', str(datetime.timedelta(seconds=e)))
    support_b = feature_selection(model, X_train, y_train, 'backward')

    # support_f = results[0]
    selected_f = feature_names[support_f]
    print(f"Features selected by SequentialFeatureSelector (forward): {selected_f}")

    # support_b = results[1]
    selected_b = feature_names[support_b]
    print(f"\nFeatures selected by SequentialFeatureSelector (backward): {selected_b}")

    # saving the dataframe with only the selected features, depending on the selection method

    X.drop(X.columns.difference(selected_f), axis=1, inplace=True)
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_forward_selection.csv'), index=False)

    X.drop(X.columns.difference(selected_b), axis=1, inplace=True)
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_backwards_selection.csv'), index=False)

    VIP_features = np.isin(selected_f, selected_b)
    # print(feature_names[VIP_features])
    # saving the dataframe with only the selected features shared with both the selection methods
    X.drop(X.columns.difference(selected_f[VIP_features]), axis=1, inplace=True)
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_forward_and_backward_selection.csv'), index=False)
