# Libraries:
# Data manipulation:
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from pathlib import Path

# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from xgboost import XGBClassifier
from modelling.tuning.xgboost_tuner import tuner
from modelling.reporting.classifier_report import report_model_results


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
                                    tol=None,
                                    n_jobs=-1)
    print(f'> performing feature selection. Method: {direction}')
    sfs.fit(x_tr, y_tr.values.ravel())
    print(f'sfs_{direction} fitted')
    print(f'shape ({direction}):', sfs.transform(x_tr).shape)
    support = sfs.get_support()
    return support


if __name__ == '__main__':
    # import the  tsfel dataset:
    # Todo: use the complete dataset once it's ready.
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_tsfel.csv'))
    df_agg = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    # check that both df are sorted, so that customer ids match
    assert X.loc[0, 'CustomerId'] == df_agg.loc[0, 'CustomerId'], 'CustomerId are not matching (not sorted dataframes)'
    X.drop('CustomerId', axis=1, inplace=True)
    # import the label dataset:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'))

    # X = X[:50]  # slice for debugging
    # y = y[:50]  # slice for debugging

    feature_names = np.array(X.columns)
    # print(feature_names)

    # perform the train test split:
    X_train, X_validation, X_test, y_train, y_validation, y_test = \
        train_validation_test_split(X, y, validation=True)

    # best = tuner(X_train, y_train, X_validation, y_validation, cross_validation=5)

    # define the model:
    model = XGBClassifier(objective="binary:logistic", random_state=42, n_jobs=-1)

    # # multiprocessing to perform feature selection simultaneously...doesn't work
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(feature_selection, model, X_train, y_train, direction)
    #                for direction in ['forward', 'backward']]
    #     # wait for all the futures to finish
    #     results = [future.result() for future in futures]  # returns the selector
    #     # catch exceptions:
    #     for future in futures:
    #         if future.exception() is not None:
    #             print(future.exception())
    #             # remove the future from the list
    #             futures.remove(future)
    # print('> task submitted')

    support_f = feature_selection(model, X_train, y_train, 'forward')
    support_b = feature_selection(model, X_train, y_train, 'backward')

    # support_f = results[0]
    # print(f'shape of the selected features (forward):', sfs_f.transform(X_train).shape)
    # support_f = sfs_f.get_support()
    selected_f = feature_names[support_f]
    print(f"Features selected by SequentialFeatureSelector (forward): {selected_f}")

    # support_b = results[1]
    # print(f'shape of the selected features (forward):', sfs_b.transform(X_train).shape)
    # support_b = sfs_b.get_support()
    selected_b = feature_names[support_b]
    print(f"\nFeatures selected by SequentialFeatureSelector (backward): {selected_b}")

    # fit the model:
    print('> fitting xgboost model')
    model.fit(X_train, y_train.values.ravel())

    # predict:
    y_pred = model.predict(X_test)

    # evaluate:
    report_model_results(model, X_train, X_test, y_test, y_pred, "Time series enriched RFM model (wrapper)", save=True)

    # saving the dataframe with only the selected features, depending on the selection method

    X.drop(X.columns.difference(selected_f), axis=1, inplace=True)
    X.to_csv(Path('..', '..', 'data', f'FS_forward_timeseries.csv'), index=False)

    X.drop(X.columns.difference(selected_b), axis=1, inplace=True)
    X.to_csv(Path('..', '..', 'data', f'FS_backward_timeseries.csv'), index=False)

    VIP_features = feature_names[np.isin(selected_f, selected_b)]
    # print(feature_names[VIP_features])
    # saving the dataframe with only the selected features shared with both the selection methods
    X.drop(X.columns.difference(VIP_features), axis=1, inplace=True)
    X.to_csv(Path('..', '..', 'data', f'FS_shared_timeseries.csv'), index=False)
