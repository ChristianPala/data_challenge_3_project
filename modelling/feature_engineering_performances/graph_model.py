# Libraries:
# data manipulation:
import pandas as pd
from pathlib import Path

# modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from modelling.tuning.xgboost_tuner import tune_xgboost
from modelling.reporting.classifier_report import report_model_results

# Functions:
if __name__ == '__main__':
    # read the graph datasets:
    X_train = pd.read_csv(Path('../..', 'data', 'customer_graph_train_centrality.csv'))
    X_val = pd.read_csv(Path('../..', 'data', 'customer_graph_val_centrality.csv'))
    X_test = pd.read_csv(Path('../..', 'data', 'customer_graph_test_centrality.csv'))

    # read the aggregated dataset:
    X = pd.read_csv(Path('../..', 'data', 'online_sales_dataset_agg.csv'))
    y = X['CustomerChurned']
    X.drop(columns=['CustomerChurned'], inplace=True)

    # add the RFM features from the aggregated dataset:
    X_train = X_train.merge(X[['CustomerId', 'Recency', 'NumberOfPurchases', 'TotalSpent',
                               'TotalQuantity', 'Country']], on='CustomerId')

    X_val = X_val.merge(X[['CustomerId', 'Recency', 'NumberOfPurchases', 'TotalSpent',
                           'TotalQuantity', 'Country']], on='CustomerId')

    X_test = X_test.merge(X[['CustomerId', 'Recency', 'NumberOfPurchases', 'TotalSpent',
                             'TotalQuantity', 'Country']], on='CustomerId')

    # drop the customer id:
    X_train.drop('CustomerId', axis=1, inplace=True)
    X_val.drop('CustomerId', axis=1, inplace=True)
    X_test.drop('CustomerId', axis=1, inplace=True)

    _, _, _, y_train, y_val, y_test = \
        train_validation_test_split(X, y, validation=True)

    # tune the model with bayesian optimization:
    best_parameters = tune_xgboost(X_train, y_train, X_val, y_val, cross_validation=5)

    # print the best parameters:
    print('Best parameters:')
    print(best_parameters)

    # train the model with the best parameters:
    model = XGBClassifier(**best_parameters, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_predicted = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, X_test, y_test, y_predicted, "Graph enriched RFM model", save=True)
