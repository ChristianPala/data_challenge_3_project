# Libraries:
# data manipulation:
import pandas as pd
from pathlib import Path
import pickle
import tabulate
# modelling:
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from tuning.xgboost_tuner import tuner

# Functions:
if __name__ == '__main__':
    # read the page rank dataset:
    X = pd.read_csv(Path('..', 'data', 'customer_pagerank.csv'))

    # drop the unnamed column:
    X.drop(columns=['Unnamed: 0'], inplace=True)

    # get the target from the aggregated dataset:
    y = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))['CustomerChurned']

    # add the features of the base model (remove to see the effect of pagerank in isolation):
    X = pd.concat([X, pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))
    [['NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country', 'Recency']]], axis=1)

    # train test split:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)

    # tune the model with bayesian optimization:
    best_parameters = tuner(X_train, y_train, X_test, y_test, max_evaluations=1000, early_stopping=30)

    # print the best parameters:
    print('Best parameters:')
    print(best_parameters)

    # train the model with the best parameters:
    model = XGBClassifier(**best_parameters, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_predicted = model.predict(X_test)

    # print the classification report and the f1 score:
    print(classification_report(y_test, y_predicted))
    print(f"Graph enhanced model has an f-score of: {f1_score(y_test, y_predicted):.3f}")

    # print the feature importance:
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    # sort by importance:
    importance.sort_values(by='importance', ascending=False, inplace=True)
    # print the feature importance with tabulate:
    print(tabulate.tabulate(importance, headers='keys', tablefmt='psql'))

