# Libraries:

# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from modelling.reporting.classifier_report import report_model_results
from modelling.tuning.xgboost_tuner import tuner


# Functions:
def evaluate_csv(file_path: Path, file_name: str, fast: bool = False) -> None:
    """
    Evaluate the performance of the model on the dataset with the features selected by the csv file.
    @param file_path: the path to the csv file.
    @param file_name: the name of the feature selection method.
    @param fast: whether to use fast tuning or not.
    :return: saves the results of the model in the plots folder.
    """
    # read the aggregated dataset:
    X = pd.read_csv(file_path, index_col=0)

    # get the labels from tsfel:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # train test split:
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y, validation=True)

    # tune the model, fast tuning since the search space is so large, yields good results:
    best_params = tuner(X_train, y_train, X_val, y_val, fast=fast)

    # define the model:
    model = XGBClassifier(**best_params, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_pred = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, X_test, y_test, y_pred, file_name, save=True)


# Driver:
if __name__ == '__main__':
    # paths:
    full_features_path = Path('..', '..', 'data', 'online_sales_dataset_for_fs.csv')
    variance_threshold_path = Path('..', '..', 'data', 'online_sales_dataset_fs_variance_threshold.csv')
    mutual_info_path = Path('..', '..', 'data', 'online_sales_dataset_fs_mutual_information.csv')
    forward_selection_path = Path('..', '..', 'data', 'online_sales_dataset_fs_forward_selection.csv')
    backward_selection_path = Path('..', '..', 'data', 'online_sales_dataset_fs_backwards_selection.csv')
    recursive_elimination_path = Path('..', '..', 'data', 'online_sales_dataset_fs_rfe.csv')

    # evaluate the models:
    # evaluate_csv(full_features_path, 'all_fe_features', fast=True)
    # evaluate_csv(variance_threshold_path, 'variance_threshold_fs', fast=True)
    evaluate_csv(mutual_info_path, 'mutual_information_fs', fast=True)
    # evaluate_csv(forward_selection_path, 'forward_selection_fs', fast=True)
    # evaluate_csv(backward_selection_path, 'backward_selection_fs', fast=True)
    # evaluate_csv(recursive_elimination_path, 'recursive_elimination_100_fs', fast=True)


