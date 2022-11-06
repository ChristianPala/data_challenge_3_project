# Auxiliary library to combine the results from the wrapper methods
# based on feature commonality and performance with the tuned XGB model
# with embedded methods based on feature importance.

from pathlib import Path

# Libraries:
# Data manipulation:
import pandas as pd
from modelling.feature_selection_performances.evaluate_features_selection import evaluate_csv


def main():
    # load the dataset for feature selection:
    X = pd.read_csv(Path('data', 'online_sales_dataset_for_fs.csv'))

    # In this version we used the result of the forward selection with cross-validation and automatic
    # number of features selected:
    full_features_path = Path('data', 'online_sales_dataset_for_fs.csv')
    print('> searching for the best features')
    evaluate_csv(full_features_path, 'all_fe_features', fast=True)
    find_features = pd.read_csv(Path('data', f'feature_importance_all_fe_features.csv'))
    best_features = find_features[find_features['importance'] > 0.005]
    # select the features above:
    features = best_features.loc[:, 'feature'].to_list()
    features.insert(0, 'CustomerId')

    # keep only the features above in X:
    X = X[features]

    # save the dataset:
    X.to_csv(Path('data', 'online_sales_dataset_for_dr.csv'), index=False)


# Driver:
if __name__ == '__main__':
    main()
