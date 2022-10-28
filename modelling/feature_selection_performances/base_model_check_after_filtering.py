# Libraries:

# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
# Tuning:
from modelling.tuning.xgboost_tuner import tune_xgboost


# Driver:
if __name__ == '__main__':

    # load the dataset:
    X = pd.read_csv(Path('../..', 'data', 'online_sales_dataset_for_fs_mutual_information.csv'), index_col=0)
    y = pd.read_csv(Path('../..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # Select only the RFM features of the base model:
    X = X[['Recency', 'TotalSpent', 'TotalQuantity', 'NumberOfPurchases']]

    # train test split with validation set:
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y, validation=True)

    # tune the model:
    best_params = tune_xgboost(X_train, y_train, X_val, y_val)

    print("The best parameters are: ")
    print(best_params)

    # define the model:
    model = XGBClassifier(**best_params, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # evaluate the model:
    y_predicted = model.predict(X_test)

    # f-score:
    f_score = f1_score(y_test, y_predicted)

    """
    The f-score is comparable to the base model, no inconsistency found.
    """





