# implement feature selection with the SHAP package:
import numpy as np
# Libraries:
import pandas as pd
from pathlib import Path
import shap
from xgboost import XGBClassifier

from modelling.data_splitting.train_val_test_splitter import train_validation_test_split


# Functions:
def main() -> None:
    """
    Performs feature selection with the SHAP package.
    :return: None
    """

    # import the RFM dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    y = df['CustomerChurned']

    # keep only the RFM features:
    X = df[['Recency', 'NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]

    # split the dataset into train, validation and test:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)
    # get the features and the target:

    # train the model:
    model = XGBClassifier(objective="binary:logistic", n_estimators=500, random_state=42)

    model.fit(X_train, y_train)

    # calculate the shap values:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # plot the shap values:
    shap.summary_plot(shap_values, X_train, plot_type="bar")

    # other plots:
    shap.dependence_plot("Recency", shap_values, X_train)
    shap.dependence_plot("NumberOfPurchases", shap_values, X_train)
    shap.dependence_plot("TotalSpent", shap_values, X_train)
    shap.dependence_plot("TotalQuantity", shap_values, X_train)
    shap.dependence_plot("Country", shap_values, X_train)

    # get the shap values for the test set:
    shap_values_test = explainer.shap_values(X_test)

    # plot the shap values for the test set:
    shap.summary_plot(shap_values_test, X_test, plot_type="bar")

    # other plots:
    shap.dependence_plot("Recency", shap_values_test, X_test)
    shap.dependence_plot("NumberOfPurchases", shap_values_test, X_test)
    shap.dependence_plot("TotalSpent", shap_values_test, X_test)
    shap.dependence_plot("TotalQuantity", shap_values_test, X_test)
    shap.dependence_plot("Country", shap_values_test, X_test)

    # save the shap values:
    shap_values_df = pd.DataFrame(shap_values, columns=X_train.columns)
    shap_values_df.to_csv(Path('..', '..', 'data', 'shap_values.csv'))


# Driver:
if __name__ == '__main__':
    main()
