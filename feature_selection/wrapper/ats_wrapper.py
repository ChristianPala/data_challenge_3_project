# Libraries:
# Data manipulation:
import os
import pandas as pd
from pathlib import Path

# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from modelling.tuning.xgboost_tuner import tuner
from modelling.reporting.classifier_report import report_model_results


if __name__ == '__main__':
    # import the  tsfel dataset:
    curr_dir = os.getcwd()  # breaks my balls when I use '..' twice, and only here, not in the other files
    X = pd.read_csv(Path(curr_dir, '..', 'data', 'online_sales_dataset_tsfel.csv'))
    # import the label dataset:
    y = pd.read_csv(Path(curr_dir, '..', 'data', 'online_sales_labels_tsfel.csv'))

    # remove all columns with NaN values:
    X = X.dropna(axis=1)

    # perform the train test split:
    X_train, X_validation, X_test, y_train, y_validation, y_test = \
        train_validation_test_split(X, y, validation=True)

    # tune xgboost for time series data:
    best = tuner(X_train, y_train, X_validation, y_validation, cross_validation=5)

    # define the model:
    model = XGBClassifier(**best, objective="binary:logistic", random_state=42, n_jobs=-1)

    # model = KNeighborsClassifier(n_neighbors=5)

    sfs = SequentialFeatureSelector(model, direction='forward', n_features_to_select='auto', tol=None, n_jobs=-1)
    print('performing feature selection')
    sfs.fit(X_train, y_train.values.ravel())

    print('transforming')
    print(sfs.transform(X_train).shape)

    # fit the model:
    print('fitting xgboost model')
    model.fit(X_train, y_train.values.ravel())

    # predict:
    y_pred = model.predict(X_test)

    # evaluate:
    report_model_results(model, X_train, X_test, y_test, y_pred, "Time series enriched RFM model (wrapper)", save=True)
