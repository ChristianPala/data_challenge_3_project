# Test the performance of the model after recursive feature elimination:

# Libraries:

# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from xgboost import XGBClassifier
from data_splitting.train_val_test_splitter import train_validation_test_split

# Tuning:
from tuning.xgboost_tuner import tuner

# Evaluation:
from reporting.classifier_report import report_model_results


# Driver:
if __name__ == '__main__':

    # load the dataset:
    X = pd.read_csv(Path('..', 'data', 'online_sales_dataset_for_fs_rfe.csv'), index_col=0)
    y = pd.read_csv(Path('..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # train test split with validation set:
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y, validation=True)

    # tune the model:
    best_params = tuner(X_train, y_train, X_val, y_val)

    print("The best parameters are: ")
    print(best_params)

    # define the model:
    model = XGBClassifier(**best_params, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # evaluate the model:
    y_predicted = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, X_test, y_test, y_predicted, "recursive_elimination_100_model", save=True)

