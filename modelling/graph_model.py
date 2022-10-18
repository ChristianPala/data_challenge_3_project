# Libraries:
# data manipulation:
import pandas as pd
from pathlib import Path

# modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from tuning.xgboost_tuner import tuner
from reporting.classifier_report import report_model_results

# Functions:
if __name__ == '__main__':
    # read the pagerank dataset:
    X = pd.read_csv(Path('..', 'data', 'customer_pagerank.csv'))

    # read the deepwalk dataset:
    # X_deepwalk = pd.read_csv(Path('..', 'data', 'customer_deepwalk_embeddings.csv'))

    # merge the datasets on CustomerId:
    # X = X.merge(X_deepwalk, on='CustomerId')

    # remove the customer id column:
    X.drop('CustomerId', axis=1, inplace=True)

    # get the target from the aggregated dataset:
    y = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))['CustomerChurned']

    # add the features of the base model (remove to see the effect of pagerank in isolation):
    X = pd.concat([X, pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))
    [['NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country', 'Recency']]], axis=1)

    # train validation test split:
    X_train, X_validation, X_test, y_train, y_validation, y_test = train_validation_test_split(X, y, validation=True)

    # tune the model with bayesian optimization:
    best_parameters = tuner(X_train, y_train, X_validation, y_validation, cross_validation=5)

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
    report_model_results(model, X_train, y_test, y_predicted, "Graph enriched RFM model", save=True)
