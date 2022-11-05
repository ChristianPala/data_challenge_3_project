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
    # load the nlp features:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_nlp_for_fs.csv'), index_col=0)

    # get the target from the aggregated dataset:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))['CustomerChurned']

    # load the aggregated dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'), index_col=0)

    # match the two on the customer id:
    X = df.merge(X, on='CustomerId')

    # exclude the target and the description:
    X = X.drop(['CustomerChurned', 'Description'], axis=1)

    # split the train set into train and validation sets:
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y, validation=True)

    # tune xgboost for the base plus nlp data:
    best_params = tuner(X_train, y_train, X_val, y_val, fast=True)
    # fast=True means we do 2 cross validations, we needed this setting to speed up the process.

    # save the best parameters:
    pd.DataFrame(best_params, index=[0]).to_csv(Path('..', '..', 'data', 'best_params', 'nlp_model.csv'),
                                                index=False)

    # define the model:
    model = XGBClassifier(**best_params, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # evaluate the model:
    y_predicted = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, X_test, y_test, y_predicted, "NLP model", save=True)
