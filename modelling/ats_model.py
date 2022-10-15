# Libraries:
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV

matplotlib.use('TkAgg')


if __name__ == '__main__':
    # import the  tsfel dataset:
    df_ts = pd.read_csv(Path('..', 'data', 'online_sales_dataset_tsfel.csv'))
    # import the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))
    # drop the first row:
    df_ts = df_ts.drop(df_ts.index[0])

    # remove all columns with NaN values:
    df_ts = df_ts.dropna(axis=1)

    # add the features we used in the base model:
    df_ts["NumberOfPurchases"] = df_agg["NumberOfPurchases"].astype(int)
    df_ts["TotalSpent"] = df_agg["TotalSpent"]
    df_ts["TotalQuantity"] = df_agg["TotalQuantity"]
    df_ts["Country"] = df_agg["Country"]

    # add the target variable:
    df_ts["CustomerChurned"] = df_agg["CustomerChurned"]

    # perform the train test split:
    X_train, X_test, y_train, y_test = \
        train_test_split(df_ts.drop('CustomerChurned', axis=1),
                         df_ts['CustomerChurned'], test_size=0.2, random_state=42)

    # hyperparameter tuning on the number of trees:
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=100)]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators}

    # Random search of parameters, using 3-fold cross validation

    rf = RandomForestClassifier()
    metric = make_scorer(f1_score)

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   n_iter=100, cv=3, scoring=metric, random_state=42, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(X_train, y_train)

    # print the best parameters:w
    print(rf_random.best_params_)

    # initialize model with the best parameters:
    model = RandomForestClassifier(n_estimators=rf_random.best_params_['n_estimators'], random_state=42)

    # train the model:
    model.fit(X_train, y_train)

    # predict:
    y_pred = model.predict(X_test)

    # evaluate:
    print(classification_report(y_test, y_pred))
    print(f1_score(y_test, y_pred))

    # visualize features importance, sorted by importance, cut off at 0.5 threshold:
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    # sort by importance:
    importance.sort_values(by='importance', ascending=False, inplace=True)

    top_n = 10
    importance = importance.iloc[:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar(importance['feature'][0:9], importance['importance'][0:9])
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()
