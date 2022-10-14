# Libraries:
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay
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
    X_test.drop(['CustomerId', 'Description'], axis=1, inplace=True)

    # train the best random forest model found for the best case:
    model = RandomForestClassifier(n_estimators=500)
    model.fit(X_train, y_train.values.ravel())

    # evaluate the model on the f1-score:
    y_pred = model.predict(X_test)
    print(f"Model f1-score: {f1_score(y_test, y_pred):.3f}")

    # print the feature importance for each feature:
    for feature, importance in zip(X_train.columns, model.feature_importances_):
        print(f"{feature}: {importance:.3f}")

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

    plt.show()



