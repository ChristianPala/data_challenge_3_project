# Tune the random forest model:
# read this kaggle article as a basis:
# https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook
#
# # Libraries:
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')

# Functions:
if __name__ == '__main__':
    # read the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # select the features: number of purchases, total price spent, total quantity ordered and country:
    X = df_agg[['NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]
    y = df_agg['CustomerChurned']

    # train test split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # tune the model with bayesian optimization:

    # define the search space:
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

    # print the best parameters:
    print("The best parameters are: ")
    print(best)

    # the best parameters are:
    """
    {'colsample_bytree': 0.4771911169038492, 'gamma': 0.5915603402393034, 'learning_rate': 0.813805917920331,
    'max_depth': 8, 'min_child_weight': 5, 'n_estimators': 127, 'reg_alpha': 0.538492486622712,
    'reg_lambda': 0.37144751917622304, 'subsample': 0.01129889610694531}
    """

    # train the model with the best parameters:
    model = XGBClassifier(n_estimators=127,
                          max_depth=8,
                          learning_rate=0.813805917920331,
                          min_child_weight=5,
                          gamma=0.5915603402393034,
                          subsample=0.01129889610694531,
                          colsample_bytree=0.4771911169038492,
                          reg_alpha=0.538492486622712,
                          reg_lambda=0.37144751917622304,
                          objective="binary:logistic",
                          random_state=42,
                          n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_predicted = model.predict(X_test)

    # print the classification report and the f1 score:
    print(classification_report(y_test, y_predicted))
    print(f"Tuned base model has an f-score of: {f1_score(y_test, y_predicted):.3f}")
    # 0.528

    # save the model as a pickle file:
    with open(Path('..', 'models', 'xgboost_base_model_tuned.pkl'), 'wb') as f:
        pickle.dump(model, f)
