# Libraries:
# Data manipulation:
from pathlib import Path

import pandas as pd
# Modelling:
from xgboost import XGBClassifier

from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from modelling.reporting.classifier_report import report_model_results


def main():
    # read the aggregated dataset:
    df_agg = pd.read_csv(Path('data', 'online_sales_dataset_agg.csv'))

    # select the features: number of purchases, total price spent, total quantity ordered and country:
    X = df_agg[['NumberOfPurchases', 'NumberOfProducts', 'TotalSpent',
                'TotalQuantity', 'Country']]
    y = df_agg['CustomerChurned']

    # train test split:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)

    # define the model:
    model = XGBClassifier(objective="binary:logistic", random_state=42)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_pred = model.predict(X_test)

    # report the results:
    report_model_results(model, X_train, X_test, y_test, y_pred, "Base model without tuning", save=True)


if __name__ == '__main__':
    main()
