# Library to perform exhaustive feature selection using brute force.

# Data manipulation:
import pandas as pd
from pathlib import Path

# Feature selection:
# exhaustive feature selection:
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# Modelling:
from xgboost import XGBClassifier
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
def main():
    # load the mutual information filtered dataset:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_mutual_information_0.001.csv'), index_col=0)

    # get the time series labels from the dataset:
    y = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # train test split:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)

    # create the model:
    model = XGBClassifier(objective="binary:logistic", random_state=42, n_jobs=-1)

    # create the EFS object:
    efs = EFS(model, min_features=1, max_features=1, scoring='f1',
              cv=2, n_jobs=-1, print_progress=True, fixed_features=())

    # X.columns.get_loc('1_Area under the curve')

    # fit the EFS object to the dataset:
    efs.fit(X_train, y_train)

    # get the feature names selected by EFS:
    efs_features = X.columns[list(efs.best_idx_)]
    print(efs_features)

    # keep only the features selected by EFS:
    X = X[efs_features]

    # save the dataset:
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_fs_exhaustive.csv'))

    # print the features selected by EFS:
    print(efs.best_feature_names_)
    # 1 feature: average days between purchases area under the curve.

    # Complementary:
    # 3 EigenvectorCentrality_country
    # 4 69
    # 5 11

    # Similar results to automatic feature selection on the mutual information filtered dataset.


# Driver:
if __name__ == '__main__':
    main()
