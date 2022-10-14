# Tune the random forest model:
#
# # Libraries:
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score, \
    average_precision_score, precision_recall_curve, f1_score, PrecisionRecallDisplay, RocCurveDisplay, \
    ConfusionMatrixDisplay

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.use('TkAgg')


if __name__ == '__main__':
    # read the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # select the features: number of purchases, total price spent, total quantity ordered and country:
    X = df_agg[['NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]
    y = df_agg['CustomerChurned']

    # train test split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # tune the model:
    # define the model:
    model = RandomForestClassifier(random_state=42)
    # define the hyperparameters:
    n_estimators = [700, 800, 900, 1000]
    max_depth = [7, 8, 9, 10]
    min_samples_split = [1, 2, 3, 4]
    min_samples_leaf = [7, 8, 9, 10]
    # create the grid:
    grid = dict(n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    # define the evaluation procedure:
    cv = StratifiedKFold(n_splits=3, random_state=42)
    # define the model evaluation metric:
    metric = make_scorer(f1_score)
    # define the grid search:
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=metric, error_score=0)
    # fit the grid search in parallel, since it's a long process:

    grid_result = grid_search.fit(X_train, y_train)
    # summarize the results:
    print(f"Best: {grid_result.best_score_:.3f} using {grid_result.best_params_}")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f"{mean:.3f} ({stdev:.3f}) with: {param}")
    # evaluate the best model:
    best_model = grid_result.best_estimator_
    # Best: 0.454 using {'max_depth': 8, 'min_samples_leaf': 2, 'min_samples_split': 100, 'n_estimators': 800}
    # optimized on the f-score.
    # best_model = RandomForestClassifier(max_depth=8, min_samples_leaf=2, min_samples_split=100, n_estimators=800)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"f1 score: {f1_score(y_test, y_pred):.3f}")
    print(f"Average precision score: {average_precision_score(y_test, y_pred):.3f}")
    print(f"Classification report: {classification_report(y_test, y_pred)}")

    # plot the confusion matrix:
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                     display_labels=best_model.classes_)
    display.plot(cmap=plt.cm.Blues)
    plt.show()
    display.figure_.savefig(Path('..', 'plots', 'confusion_matrix_base_model_tuned.png'))

    # plot the precision recall curve:
    display = PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test, name="Random Forest")
    display.ax_.set_title(f'Precision-Recall curve base model tuned')
    plt.show()
    display.figure_.savefig(Path('..', 'plots', 'precision_recall_curve_model_tuned.png'))

    # plot the ROC curve:
    display = RocCurveDisplay.from_estimator(best_model, X_test, y_test, name="Random Forest")
    display.ax_.set_title(f'ROC curve base model tuned')
    plt.title(f'ROC curve base model tuned')
    plt.show()
    plt.savefig(Path('..', 'plots', 'roc_curve_base_model_tuned.png'))

    # save the model:
    with open(Path('..', 'models', 'rf_base_model_tuned.pkl'), 'wb') as f:
        pickle.dump(best_model, f)





