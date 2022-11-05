# Libraries:
# data manipulation:
import pandas as pd
from pathlib import Path

# modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from modelling.tuning.xgboost_tuner import tuner
from modelling.reporting.classifier_report import report_model_results

# Functions:
if __name__ == '__main__':
    # read the graph datasets:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_graph_for_fs.csv'), index_col=0)

    # get the target from the aggregated dataset:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))['CustomerChurned']

    X_train, X_val, X_test, y_train, y_val, y_test = \
        train_validation_test_split(X, y, validation=True)

    # tune the model with bayesian optimization:
    best_parameters = tuner(X_train, y_train, X_val, y_val, fast=True)

    # save the best parameters:
    pd.DataFrame(best_parameters, index=[0]).to_csv(Path('..', '..', 'data', 'best_params', 'graph_model.csv'),
                                                    index=False)

    # train the model with the best parameters:
    model = XGBClassifier(**best_parameters, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_predicted = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, X_test, y_test, y_predicted, "Graph model", save=True)
