# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from xgboost import XGBClassifier
from modelling.tuning.xgboost_tuner import tuner
from modelling.reporting.classifier_report import report_model_results


if __name__ == '__main__':
    # import the  tsfel dataset:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_tsfel_for_fs.csv'), index_col=0)
    # import the label dataset:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # perform the train test split:
    X_train, X_validation, X_test, y_train, y_validation, y_test = \
        train_validation_test_split(X, y, validation=True)

    # tune xgboost for time series data:
    best = tuner(X_train, y_train, X_validation, y_validation, fast=True)

    # define the model:
    model = XGBClassifier(**best, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train.values.ravel())

    # predict:
    y_pred = model.predict(X_test)

    # evaluate:
    report_model_results(model, X_train, X_test, y_test, y_pred, "Time series model", save=True)
