# # Libraries:
# data manipulation:
import pandas as pd
from pathlib import Path

# modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from modelling.tuning.xgboost_tuner import tuner
from modelling.reporting.classifier_report import report_model_results

def main():
    # read the aggregated dataset:
    df_agg = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    # select the features for the model:
    X = df_agg[['NumberOfPurchases', 'NumberOfProducts', 'TotalSpent', 'TotalQuantity', 'Country']]
    y = df_agg['CustomerChurned']

    # train test split:
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y, validation=True)

    # tune the model with bayesian optimization:
    best_parameters = tuner(X_train, y_train, X_val, y_val, fast=True)
    # fast=True means we do 2 cross validations, we needed this setting to speed up the process.

    # print the best parameters:
    print('Best parameters:')
    print(best_parameters)

    # save the best parameters:
    pd.DataFrame(best_parameters, index=[0]).to_csv(Path('..', '..', 'data', 'best_params', 'base_model.csv'),
                                                    index=False)

    # train the model with the best parameters:
    model = XGBClassifier(**best_parameters, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_predicted = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, X_test, y_test, y_predicted, "Tuned RFM model", save=True)


# Functions:
if __name__ == '__main__':
    main()
