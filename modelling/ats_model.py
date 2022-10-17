# Libraries:
import pandas as pd
from pathlib import Path

from sklearn.metrics import classification_report, f1_score
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from xgboost import XGBClassifier
from tuning.xgboost_tuner import tuner

from tabulate import tabulate
import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':
    # import the  tsfel dataset:
    df_ts = pd.read_csv(Path('..', 'data', 'online_sales_dataset_tsfel.csv'))
    # import the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))
    # drop the first row:
    df_ts = df_ts.drop(df_ts.index[0])

    # remove all columns with NaN values:
    df_ts = df_ts.dropna(axis=1)

    # add the features we used in the base model:
    # df_ts["NumberOfPurchases"] = df_agg["NumberOfPurchases"].astype(int)
    # df_ts["TotalSpent"] = df_agg["TotalSpent"]
    # df_ts["TotalQuantity"] = df_agg["TotalQuantity"]
    # df_ts["Country"] = df_agg["Country"]
    # df_ts["Recency"] = df_agg["Recency"]

    # add the target variable:
    df_ts["CustomerChurned"] = df_agg["CustomerChurned"]

    # perform the train test split:
    X_train, X_test, X_validation, y_validation, y_train, y_test = \
        train_validation_test_split(df_ts.drop('CustomerChurned', axis=1), df_ts['CustomerChurned'], validation=True)

    # tune xgboost for time series data:
    best = tuner(X_train, y_train, X_validation, y_validation, cross_validation=5)
    print('Best parameters:')
    print(best)

    # define the model:
    model = XGBClassifier(**best, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train.values.ravel())

    # predict:
    y_pred = model.predict(X_test)

    # evaluate:
    print(classification_report(y_test, y_pred))
    print(f"The f1 score of the time series model is : {f1_score(y_test, y_pred):.3f}")

    # print feature importance:
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    # sort by importance:
    importance.sort_values(by='importance', ascending=False, inplace=True)

    # print the feature importance in a table:
    print(tabulate(importance, headers='keys', tablefmt='psql'))

    # save the feature importance in a csv file:
    importance.to_csv(Path('..', 'data', 'feature_importance_ts.csv'), index=False)
