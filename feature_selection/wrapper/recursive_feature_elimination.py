# Wrapper method for feature selection using RFE:
import numpy as np
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold
from tabulate import tabulate

# Feature selection:
from sklearn.feature_selection import RFECV

# Modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split

# Driver:
if __name__ == '__main__':
    # get the dataset from the mutual information filtering:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_mutual_information.csv'), index_col=0)

    # get the time series labels from the dataset:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # ravel the target variable, required by sklearn:
    y = np.ravel(y)

    # separate the features from the target variable:
    y = X['target']
    X = X.drop('target', axis=1)

    # train test split:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)

    # create the model:
    model = XGBClassifier()

    # create the cross validation object:
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

    # create the RFE object with automatic feature number selection and cross validation:
    rfe = RFECV(estimator=model, step=1, cv=cv, scoring='f1', n_jobs=-1)

    # fit the RFE object to the dataset:
    rfe.fit(X_train, y_train)

    # get the features selected by RFE:
    rfe_features = X.columns[rfe.support_]

    # keep only the features selected by RFE:
    X = X[rfe_features]

    # save the dataset:
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_rfe.csv'))

    # print the features selected by RFE:
    print(tabulate([rfe_features], headers='keys', tablefmt='psql'))

    # print the ranking of the features:
    print(tabulate([rfe.ranking_], headers='keys', tablefmt='psql'))

