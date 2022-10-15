# Libraries:
import matplotlib
import pandas as pd
from pathlib import Path

from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tabulate import tabulate

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

    # tune xgboost for time series data:
    space = {
        'n_estimators': hp.choice('n_estimators', range(100, 1000)),
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'learning_rate': hp.uniform('learning_rate', 0.01, 1),
        'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
        'gamma': hp.uniform('gamma', 0.01, 1),
        'subsample': hp.uniform('subsample', 0.01, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0.01, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, 1)
    }


    # define the objective function:
    def objective(space):
        model = XGBClassifier(n_estimators=space['n_estimators'],
                              max_depth=space['max_depth'],
                              learning_rate=space['learning_rate'],
                              min_child_weight=space['min_child_weight'],
                              gamma=space['gamma'],
                              subsample=space['subsample'],
                              colsample_bytree=space['colsample_bytree'],
                              reg_alpha=space['reg_alpha'],
                              reg_lambda=space['reg_lambda'],
                              objective="binary:logistic",
                              early_stopping_rounds=10,
                              eval_metric="aucpr",
                              random_state=42,
                              n_jobs=-1)

        evaluation = [(X_train, y_train), (X_test, y_test)]

        # fit the model, the metric is the f-1 score:
        model.fit(X_train, y_train, eval_set=evaluation, verbose=False)

        # predict:
        y_pred = model.predict(X_test)

        # calculate the f1 score:
        f1 = f1_score(y_test, y_pred)

        # return the negative f1 score:
        return {'loss': -f1, 'status': STATUS_OK}


    # define the trials object:
    trials = Trials()

    # run the optimization:
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

    # if a parameter is 0, filter it out:
    best = {k: v for k, v in best.items() if v != 0}

    # define the model:
    model = XGBClassifier(**best, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train.values.ravel())

    # predict:
    y_pred = model.predict(X_test)

    # evaluate:
    print(classification_report(y_test, y_pred))
    print(f"The f1 score of the time series model is : {f1_score(y_test, y_pred):.3f}")

    # print feature importance:
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    # sort by importance:
    importance.sort_values(by='importance', ascending=False, inplace=True)

    # print the feature importance in a table:
    print(tabulate(importance, headers='keys', tablefmt='psql'))
