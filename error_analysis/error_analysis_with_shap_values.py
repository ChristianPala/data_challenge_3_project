# implement feature selection with the SHAP package:
# Libraries:
import numpy as np
# Data manipulation:
import pandas as pd
from pathlib import Path
# Error analysis:
import shap
# Modelling:
# Plotting:
import matplotlib.pyplot as plt
# evaluate the model:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from modelling.tuning.xgboost_tuner import tuner

from modelling.data_splitting.train_val_test_splitter import train_validation_test_split

def main():
    # load the dataset:
    X = pd.read_csv(Path('..', 'data', 'online_sales_dataset_for_dr.csv'), index_col=0)

    # load the labels:
    y = pd.read_csv(Path('..', 'data', 'online_sales_labels_tsfel.csv'),
                    index_col=0)['CustomerChurned']

    # shorten the column names for readability:
    X.columns = [col.replace('DeepwalkEmbedding', 'DWEmb') for col in X.columns]
    X.columns = [col.replace('AverageDaysBetweenPurchase', 'AvgDaysBtwBuy') for col in X.columns]
    X.columns = [col.replace('TotalSpent', 'TotSpent') for col in X.columns]
    X.columns = [col.replace('CustomerGraph', 'CIdGrph') for col in X.columns]
    X.columns = [col.replace('MeanCoefficient', 'MeanCoeff') for col in X.columns]

    # split the dataset, since the model has been validated on the validation set, we
    # will use the test set for the error analysis:
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(X, y, validation=True)

    # tune the model:
    best_params = tuner(X_train, y_train, X_val, y_val)

    # instantiate the model:
    model = XGBClassifier(**best_params, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # get the predictions:
    y_pred = model.predict(X_test)

    # evaluate the model:
    print("Model evaluation:")
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print(f'Precision: {precision_score(y_test, y_pred):.2f}')
    print(f'Recall: {recall_score(y_test, y_pred):.2f}')
    print(f'F1 score: {f1_score(y_test, y_pred):.2f}')

    # Train analysis:
    # initialize the shapley values:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # error analysis:
    shap.summary_plot(shap_values, X_test.values, feature_names=X_test.columns, show=False)

    # save the plot:
    plt.savefig(Path('..', 'plots', 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # get the index of the false negatives:
    fn_idx = np.where((y_test == 1) & (y_pred == 0))[0]

    # create path if it does not exist:
    if not Path('..', 'plots', 'error_analysis').exists():
        Path('..', 'plots', 'error_analysis').mkdir(parents=True)

    # Save the force plots of the false negatives:
    for i in fn_idx:
        shap.waterfall_plot(shap.Explanation(values=shap_values[i],
                                             base_values=explainer.expected_value, data=X_test.iloc[i],
                                             feature_names=X.columns), show=False)
        # give more space for the y-axis:
        plt.subplots_adjust(left=0.4, right=0.9, top=0.9, bottom=0.2)
        plt.savefig(Path('..', 'plots', 'error_analysis', f'fn_{i}.png'))
        plt.close()

    # plot some success case:
    # get some true positive:
    tp_idxs = np.where((y_test == 1) & (y_pred == 1))[0][0:5]
    # plot the force plot:
    for tp in tp_idxs:
        shap.waterfall_plot(shap.Explanation(values=shap_values[tp],
                                             base_values=explainer.expected_value, data=X_test.iloc[tp],
                                             feature_names=X.columns), show=False)
        # give more space for the y-axis:
        plt.subplots_adjust(left=0.4, right=0.9, top=0.9, bottom=0.2)
        plt.subplots_adjust(left=0.4, right=0.9, top=0.9, bottom=0.2)
        plt.savefig(Path('..', 'plots', 'error_analysis', f'tp_{tp}.png'))
        plt.close()

    # plot some true negatives:
    # get some true negative:
    tn_idxs = np.where((y_test == 0) & (y_pred == 0))[0][0:5]
    # plot the force plot:
    for tn in tn_idxs:
        shap.waterfall_plot(shap.Explanation(values=shap_values[tn],
                                             base_values=explainer.expected_value, data=X_test.iloc[tn],
                                             feature_names=X.columns), show=False)
        # give more space for the y-axis:
        plt.subplots_adjust(left=0.4, right=0.9, top=0.9, bottom=0.2)
        plt.subplots_adjust(left=0.4, right=0.9, top=0.9, bottom=0.2)
        plt.savefig(Path('..', 'plots', 'error_analysis', f'tn_{tn}.png'))
        plt.close()

    # Conclusions:


# Driver:
if __name__ == '__main__':
    main()
