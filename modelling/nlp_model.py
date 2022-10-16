# Libraries:
# data manipulation:
from pathlib import Path
import pandas as pd
import tabulate

# modelling:
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix, \
    PrecisionRecallDisplay, RocCurveDisplay
from xgboost import XGBClassifier
from tuning.xgboost_tuner import tuner

# plotting:
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


if __name__ == '__main__':

    # load the train features:
    X_train = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_train.csv'))
    y_train = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_train_labels.csv'))

    # load the test features:
    X_test = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_test.csv'))
    y_test = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_test_labels.csv'))

    # drop customer id and description:
    X_train.drop(['CustomerId', 'Description'], axis=1, inplace=True)
    X_test.drop(['CustomerId', 'Description', ], axis=1, inplace=True)

    # todo: should we drop the clusters with no matches in the test set?

    # tune xgboost for the base plus nlp data:
    best_params = tuner(X_train, y_train, X_test, y_test, max_evaluations=1000, early_stopping=20)
    """
    {'colsample_bytree': 0.3278595548390614, 'gamma': 0.548702557753846, 'learning_rate': 0.6506709414891841,
     'max_depth': 1, 'min_child_weight': 3, 'n_estimators': 609, 'reg_alpha': 0.8301138158484043,
     'reg_lambda': 0.2118779348144419, 'subsample': 0.19379912149955214}
    """

    print("The best parameters are: ")
    print(best_params)

    # define the model:
    model = XGBClassifier(**best_params, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # evaluate the model on the f1-score:
    y_pred = model.predict(X_test)
    print(f"Model f1-score: {f1_score(y_test, y_pred):.3f}")
    """
    Model
    f1 - score: 0.714
    """
    # print the feature importance for each feature:
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    # sort by importance:
    importance.sort_values(by='importance', ascending=False, inplace=True)
    # print the feature importance, filter out the features with importance 0:
    print(tabulate.tabulate(importance[importance['importance'] > 0], headers='keys', tablefmt='psql'))

    # plot the confusion matrix:
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                     display_labels=model.classes_)
    display.plot(cmap=plt.cm.Blues)
    display.figure_.savefig(Path('..', 'plots', 'confusion_matrix_nlp.png'))

    # plot the precision recall curve:
    display = PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name="Random Forest")
    display.ax_.set_title(f'Precision-Recall curve nlp model')
    display.figure_.savefig(Path('..', 'plots', 'precision_recall_curve_nlp_model.png'))

    # plot the ROC curve:
    display = RocCurveDisplay.from_estimator(model, X_test, y_test, name="Random Forest")
    display.ax_.set_title(f'ROC curve nlp model')
    plt.title(f'ROC curve nlp model')
    plt.savefig(Path('..', 'plots', 'roc_curve_nlp_model.png'))

    # save the feature importance:
    importance.to_csv(Path('..', 'data', 'feature_importance_nlp.csv'), index=False)



