# Libraries:
# data manipulation:
from pathlib import Path
import pandas as pd

# modelling:
from xgboost import XGBClassifier
from tuning.xgboost_tuner import tuner
from reporting.classifier_report import report_model_results

if __name__ == '__main__':

    # load the train features:
    X_train = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_train.csv'))
    y_train = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_train_labels.csv'))

    # load the test features:
    X_test = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_test.csv'))
    y_test = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_test_labels.csv'))

    # drop customer id and description:
    X_train.drop(['CustomerId', 'Description'], axis=1, inplace=True)
    X_test.drop(['CustomerId', 'Description', ], axis=1, inplace=True)

    # todo: should we drop the clusters with no matches in the test set?

    # tune xgboost for the base plus nlp data:
    best_params = tuner(X_train, y_train, X_test, y_test, max_evaluations=1000, cross_validation=5)
    """
    {'colsample_bytree': 0.3278595548390614, 'gamma': 0.548702557753846, 'learning_rate': 0.6506709414891841,
     'max_depth': 1, 'min_child_weight': 3, 'n_estimators': 609, 'reg_alpha': 0.8301138158484043,
     'reg_lambda': 0.2118779348144419, 'subsample': 0.19379912149955214}
    """

    print("The best parameters are: ")
    print(best_params)

    # define the model:
    model = XGBClassifier(**best_params, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # evaluate the model:
    y_predicted = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, y_test, y_predicted, "NLP enriched RFM model", save=True)