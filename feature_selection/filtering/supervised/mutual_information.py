# Select features using mutual information:
from pathlib import Path

import numpy as np
# Data manipulation:
import pandas as pd
# Modelling:
from sklearn.feature_selection import mutual_info_classif

from modelling.data_splitting.train_val_test_splitter import train_validation_test_split

# Global variables:
# set a threshold for the mutual information:
threshold: float = 10 ** -3  # based on the model evaluation results.


def main():
    # import the dataset for feature selection, filteres by variance threshold:
    X = pd.read_csv(Path('data',
                         f'online_sales_dataset_fs_variance_threshold_{threshold}.csv'), index_col=0)

    print(f"Incoming features: {X.shape[1]}")

    # import the target variable from the time series labels:
    y = pd.read_csv(Path('data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # ravel the target variable, required by sklearn:
    y = np.ravel(y)

    # split the dataset into train and test:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)

    # calculate the mutual information:
    # mutual information ranges from 0 to 1,
    # 0 means no mutual information,
    # 1 means perfect mutual information
    mi = mutual_info_classif(X_train, y_train)

    # get the features with mutual information above the threshold:
    mi_features = list(X_train.columns[mi > threshold])
    X = X[mi_features]

    print(f"Outgoing features: {X.shape[1]}")

    # save the dataset:
    X.to_csv(Path('data', f'online_sales_dataset_fs_mutual_information_{threshold}.csv'))

    """
    incoming features: 303
    Outgoing features: 206
    """


# Driver:
if __name__ == '__main__':
    main()
