# Libraries:
from pathlib import Path
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
matplotlib.use('TkAgg')

if __name__ == '__main__':
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # select the features: number of purchases, total price spent, total quantity ordered and country:
    X = df_agg[['NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]
    y = df_agg['CustomerChurned']

    # train test split:
    _, _, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # import nlp aggregated train and test sets:
    X_train = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_train.csv'))
    X_test = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp_test.csv'))

    # drop customer id and description:
    X_train.drop(['CustomerId', 'Description'], axis=1, inplace=True)
    X_test.drop(['CustomerId', 'Description'], axis=1, inplace=True)

    # train the best random forest model found:
    best_model = RandomForestClassifier(max_depth=8, min_samples_leaf=2, min_samples_split=100, n_estimators=800)
    best_model.fit(X_train, y_train)

    # evaluate the model on the f1-score:
    y_pred = best_model.predict(X_test)
    print(f"Best model f1-score: {f1_score(y_test, y_pred):.3f}")

    # print the feature importance:
    print(f"Feature importance: {best_model.feature_importances_}")

    # plot the feature importance:
    plt.bar(X_train.columns, best_model.feature_importances_)
    plt.title('Feature importance nlp')
    plt.show()
    plt.savefig(Path('..', 'plots', 'feature_importance_nlp.png'))

    # plot the confusion matrix:
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                     display_labels=best_model.classes_)
    display.plot(cmap=plt.cm.Blues)
    plt.show()
    display.figure_.savefig(Path('..', 'plots', 'confusion_matrix_nlp.png'))

    # plot the precision recall curve:
    display = PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test, name="Random Forest")
    display.ax_.set_title(f'Precision-Recall curve nlp model')
    plt.show()
    display.figure_.savefig(Path('..', 'plots', 'precision_recall_curve_nlp_model.png'))

    # plot the ROC curve:
    display = RocCurveDisplay.from_estimator(best_model, X_test, y_test, name="Random Forest")
    display.ax_.set_title(f'ROC curve nlp model')
    plt.title(f'ROC curve nlp model')
    plt.show()
    plt.savefig(Path('..', 'plots', 'roc_curve_nlp_model.png'))





