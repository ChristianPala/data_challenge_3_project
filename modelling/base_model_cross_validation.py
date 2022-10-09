# Libraries:

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

import pandas as pd


if __name__ == '__main__':
    # read the aggregated dataset:
    df_agg = pd.read_csv('../data/online_sales_dataset_agg.csv')

    # select the features: number of purchases, total price spent, total quantity ordered and country:
    X = df_agg[['NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]
    y = df_agg['CustomerChurned']

    # define model:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # define evaluation procedure:
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    # define the model evaluation metric:
    metric = make_scorer(accuracy_score)

    # evaluate model:
    scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)

    # print the score means and standard deviations:
    print(f"Accuracy: {scores.mean():.3f}, standard deviation: {scores.std():.3f}")

    # the base model already has a decent accuracy score.

