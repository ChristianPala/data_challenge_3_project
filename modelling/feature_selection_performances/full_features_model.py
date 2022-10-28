# Libraries:

# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from modelling.reporting.classifier_report import report_model_results
from modelling.tuning.xgboost_tuner import tune_xgboost

if __name__ == '__main__':
    # read the aggregated dataset:
    X = pd.read_csv(Path('..', '..', '..', 'data', 'online_sales_dataset_for_fs.csv'))

    # get the labels from tsfel:
    y = pd.read_csv(Path('..', '..', '..', 'data', 'online_sales_labels_tsfel.csv'))

    # train test split:
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y, validation=True)

    # tune the model:
    best_params = tune_xgboost(X_train, y_train, X_val, y_val, max_evals=1000)

    # define the model:
    model = XGBClassifier(**best_params, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_pred = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, X_test, y_test, y_pred, "full_features_model", save=True)