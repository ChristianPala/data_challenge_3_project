# auxiliary library to tune the xgboost model:

# read this kaggle article as a basis for the hyperparameter tuning:
# https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook

# Libraries:
from hyperopt import hp, Trials, fmin, STATUS_OK, tpe
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


# Functions:
def objective(space, x_train, y_train, x_test, y_test, early_stopping=10):
    """
    Objective function to be minimized.
    @param space: the hyperparameters to be tuned
    @param x_train: the training data
    @param y_train: the training labels
    @param x_test: the test data
    @param y_test: the test labels
    @param early_stopping: the number of iterations to stop early
    :return: the f1 score as a loss metric
    """
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
                          early_stopping_rounds=early_stopping,
                          eval_metric="logloss",
                          random_state=42,
                          n_jobs=-1)

    evaluation = [(x_train, y_train), (x_test, y_test)]

    # fit the model, the metric is the f-1 score:
    model.fit(x_train, y_train, eval_set=evaluation, verbose=False)

    # predict:
    y_pred = model.predict(x_test)

    # calculate the f1 score:
    f1 = f1_score(y_test, y_pred)

    # return the loss
    return {'loss': 1-f1, 'status': STATUS_OK}


def tuner(x_train, y_train, x_test, y_test, max_evaluations=100, early_stopping=10) -> dict:
    """
    Tune the xgboost model.
    @param x_train: the training data
    @param y_train: the training labels
    @param x_test: the test data
    @param y_test: the test labels
    @param max_evaluations: the maximum number of evaluations
    @param early_stopping: the number of iterations to stop early
    :return: the best hyperparameters
    """

    # define the search space, choose the parameters to tune:
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

    # define the trials object:
    trials = Trials()

    # run the optimization:
    best = fmin(fn=lambda space: objective(space, x_train, y_train, x_test, y_test, early_stopping),
                space=space, algo=tpe.suggest, max_evals=max_evaluations, trials=trials)

    # filter out the parameters that are 0:
    best = {k: v for k, v in best.items() if v != 0}

    return best
