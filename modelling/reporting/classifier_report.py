# auxiliary library to report model results:

from pathlib import Path

# Plotting:
import matplotlib
# Libraries:
# Data manipulation:
import pandas as pd
# Modelling:
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay, \
    PrecisionRecallDisplay, RocCurveDisplay, precision_score, recall_score
from tabulate import tabulate
from xgboost import XGBClassifier

matplotlib.use('TkAgg')


def report_model_results(model: XGBClassifier, x_train: pd.DataFrame, x_test: pd.DataFrame,
                         y_test: pd.Series, y_predicted: pd.Series, model_name: str,
                         plot: bool = False, save: bool = True) -> None:
    """
    Function to report the results of a model.
    @param model: the model to report the results for
    @param x_train: the training set
    @param x_test: the test set
    @param y_test: the test set labels
    @param y_predicted: the predicted labels
    @param model_name: the name of the model
    @param plot: bool: whether to plot the feature importance or not
    @param save: bool: whether to save the feature importance plot or not
    :return: None
    """
    # evaluate:
    # save the classification report in a dataframe, round the values to 3 decimals:
    report = pd.DataFrame(classification_report(y_test, y_predicted, output_dict=True)).transpose().round(3)

    # print the classification report:
    print(tabulate(report, headers='keys', tablefmt='psql'))
    # print the f1 score:
    print(f"f-score for the {model_name}: {f1_score(y_test, y_predicted): .3f}")
    # print precision:
    print(f"precision for the {model_name}: {precision_score(y_test, y_predicted): .3f}")
    # print recall:
    print(f"recall for the {model_name}: {recall_score(y_test, y_predicted): .3f}")

    # visualize initial features importance:
    importance = pd.DataFrame({'feature': x_train.columns, 'importance': model.feature_importances_})
    # sort by importance:
    importance.sort_values(by='importance', ascending=False, inplace=True)

    if plot or save:
        # if no folder plots exists, creat it:
        if not Path('plots').exists():
            Path('plots').mkdir()
        # if no folder model_name exists in plots, create it:
        if not Path('plots', model_name).exists():
            Path('plots', model_name).mkdir()

        plt.figure(figsize=(10, 6))
        plt.bar(importance['feature'], importance['importance'])
        plt.title('Embedded Feature Importance')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.savefig(Path('plots', model_name, f'feature_importance_{model_name}.png'))

        # print the feature importance in a table:
        print(tabulate(importance, headers='keys', tablefmt='presto'))

        # plot the confusion matrix:
        display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_predicted),
                                         display_labels=model.classes_)
        display.plot(cmap=plt.cm.Blues)
        display.figure_.savefig(Path('plots', model_name, f'confusion_matrix_{model_name}.png'))

        # plot the precision recall curve:
        display = PrecisionRecallDisplay.from_estimator(model, x_test, y_test, name=model_name)
        display.ax_.set_title(f'Precision-Recall curve {model_name}')
        display.figure_.savefig(Path('plots', model_name, f'precision_recall_curve_{model_name}.png'))

        # plot the ROC curve:
        display = RocCurveDisplay.from_estimator(model, x_test, y_test, name=model_name)
        display.ax_.set_title(f'ROC curve {model_name}')
        plt.title(f'ROC curve {model_name}')
        plt.savefig(Path('plots', model_name, f'roc_curve_{model_name}.png'))

        if save:
            if not Path('reports').exists():
                Path('reports').mkdir()

            if not Path('reports', model_name).exists():
                Path('reports', model_name).mkdir()

            # save the feature importance:
            importance.to_csv(Path('data', f'feature_importance_{model_name}.csv'), index=False)
            # save the classification report:
            report.to_csv(Path('reports', model_name, f'classification_report_{model_name}.csv'))
