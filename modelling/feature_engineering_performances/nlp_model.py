# Libraries:
# data manipulation:
from pathlib import Path
import pandas as pd

# modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from modelling.tuning.xgboost_tuner import tuner
from modelling.reporting.classifier_report import report_model_results

if __name__ == '__main__':

    # load the train features:
    X_train = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_train.csv'))
    y_train = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_train_labels.csv'))

    # load the test features:
    X_test = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_test.csv'))
    y_test = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_test_labels.csv'))

    # drop customer id and description:
    X_train.drop(['CustomerId', 'Description'], axis=1, inplace=True)
    X_test.drop(['CustomerId', 'Description', ], axis=1, inplace=True)

    # split the train set into train and validation sets:
    X_train, X_val, y_train, y_val = train_validation_test_split(X_train, y_train)

    # tune xgboost for the base plus nlp data:
    # best_params = tuner(X_train, y_train, X_val, y_val, cross_validation=5)
    # saved an instance to avoid re-running the tuning:

    best_params = {'colsample_bytree': 0.8643399558254448, 'gamma': 0.11916403406850484,
                   'learning_rate': 0.011208512864416056, 'max_depth': 1, 'min_child_weight': 6, 'n_estimators': 583,
                   'reg_alpha': 0.10000953068501083, 'reg_lambda': 0.6175769686746342, 'subsample': 0.4390968706966535}

    print("The best parameters are: ")
    print(best_params)

    # define the model:
    model = XGBClassifier(**best_params, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # evaluate the model:
    y_predicted = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, X_test, y_test, y_predicted, "NLP enriched RFM model", save=True)
    """
    f-score for the NLP enriched RFM model:  0.750
    """
