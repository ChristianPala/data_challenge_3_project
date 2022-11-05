# Auxiliary library to perform backwards feature selection on the Online Sales dataset.
# Library:
# Data manipulation:
import numpy as np
import pandas as pd
from pathlib import Path

from auxiliary.method_timer import measure_time
# Feature selection:
from forwards_selection import feature_selection

# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from xgboost import XGBClassifier

# Driver:
if __name__ == '__main__':
    # import the dataset:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_mutual_information_0.001.csv'))
    df_fs = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fs.csv'))

    # Drop the customer id:
    X.drop('CustomerId', axis=1, inplace=True)

    # import the label dataset:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # get the feature names:
    feature_names = np.array(X.columns)

    # perform the train test split:
    X_train, X_test, y_train, y_test = \
        train_validation_test_split(X, y)

    # define the model:
    model = XGBClassifier(objective="binary:logistic", random_state=42, n_jobs=-1)

    # perform feature selection:
    # --------------------------------------------------------------------------------------
    support_b = feature_selection(model, X_train, y_train, 'backward')
    selected_b = feature_names[support_b]
    print(f"\nFeatures selected by SequentialFeatureSelector (backward): {selected_b}")

    # Save the results:
    # --------------------------------------------------------------------------------------
    X.drop(X.columns.difference(selected_b), axis=1, inplace=True)
    # add the customer id column:
    X['CustomerId'] = df_fs['CustomerId']
    # order the columns to have the customer id as first column:
    X = X[['CustomerId'] + [col for col in X.columns if col != 'CustomerId']]
    # Save the dataset:
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_backward_selection.csv'), index=False)
