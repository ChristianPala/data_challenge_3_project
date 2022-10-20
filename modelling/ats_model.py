# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from xgboost import XGBClassifier
from tuning.xgboost_tuner import tuner
from reporting.classifier_report import report_model_results


if __name__ == '__main__':
    # import the  tsfel dataset:
    df_ts = pd.read_csv(Path('..', 'data', 'online_sales_dataset_tsfel.csv'))
    # import the label dataset:
    y = pd.read_csv(Path('..', 'data', 'online_sales_labels_tsfel.csv'))

    # remove all columns with NaN values:
    df_ts = df_ts.dropna(axis=1)

    # add the features we used in the base model:
    # df_ts["NumberOfPurchases"] = df_agg["NumberOfPurchases"].astype(int)
    # df_ts["TotalSpent"] = df_agg["TotalSpent"]
    # df_ts["TotalQuantity"] = df_agg["TotalQuantity"]
    # df_ts["Country"] = df_agg["Country"]
    # df_ts["Recency"] = df_agg["Recency"]

    # perform the train test split:
    X_train, X_validation, X_test, y_train, y_validation, y_test = \
        train_validation_test_split(df_ts, y, validation=True)

    # tune xgboost for time series data:
    best = tuner(X_train, y_train, X_validation, y_validation, cross_validation=5)

    # define the model:
    model = XGBClassifier(**best, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train.values.ravel())

    # predict:
    y_pred = model.predict(X_test)

    # evaluate:
    report_model_results(model, X_train, X_test, y_test, y_pred, "Time series enriched RFM model", save=True)
