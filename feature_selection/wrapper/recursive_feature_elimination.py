# Wrapper method for feature selection using RFE:
import numpy as np
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Feature selection:
from sklearn.feature_selection import RFE

# Modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split

# Driver:
if __name__ == '__main__':
    # get the dataset from the mutual information filtering:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fs_mutual_information.csv'), index_col=0)

    # get the time series labels from the dataset:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # ravel the target variable, required by sklearn:
    y = np.ravel(y)

    # take a small slice of the dataset for testing:
    # X = X.iloc[:500, :]
    # y = y[:500]

    # train test split:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)

    # create the model:
    model = XGBClassifier(n_estimators=500)

    # create the RFE object:
    rfe = RFE(model, n_features_to_select=100)

    # fit the RFE object to the dataset:
    rfe.fit(X_train, y_train)

    # get the features selected by RFE:
    rfe_features = X.columns[rfe.support_]

    # print the features selected by RFE:
    print(rfe_features)

    # keep only the features selected by RFE:
    X = X[rfe_features]

    # save the dataset:
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fs_rfe.csv'))
