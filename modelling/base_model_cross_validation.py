# Libraries:

from sklearn.model_selection import cross_val_score, train_test_split
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

    # split the dataset into train and test with the usual seed:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define model:
    model = RandomForestClassifier(n_estimators=500, random_state=42)

    # define evaluation procedure:
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    # define the model evaluation metric:
    metric = make_scorer(accuracy_score)

    # evaluate model:
    scores = cross_val_score(model, X_train, y_train, scoring=metric, cv=cv, n_jobs=-1)

    # print the score means and standard deviations:
    print(f"Accuracy: {scores.mean():.3f}, standard deviation: {scores.std():.3f}")

    # the base model already has a decent accuracy score.

