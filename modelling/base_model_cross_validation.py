# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split


# Driver:
if __name__ == '__main__':
    # read the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # select the features: number of purchases, total price spent, total quantity ordered and country:
    X = df_agg[['Recency', 'NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]
    y = df_agg['CustomerChurned']

    # split the dataset into train and test with the usual seed:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)

    # define model:
    model = XGBClassifier(objective="binary:logistic", n_estimators=500, random_state=42)

    # define evaluation procedure:
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

    # define the model evaluation metric:
    metric = make_scorer(f1_score)

    # evaluate model:
    scores = cross_val_score(model, X_train, y_train, scoring=metric, cv=cv, n_jobs=-1)

    # print the score means and standard deviations:
    print(f"f1 score: {scores.mean():.3f}, standard deviation: {scores.std():.3f}")

    # The base model already has a decent accuracy score, and it's quite stable.

