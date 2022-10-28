# Library to perform correlation analysis between the engineered features and the target variable.
# Data manipulation:
import pandas as pd
from pathlib import Path

# Feature selection:
from sklearn.feature_selection import VarianceThreshold

# Global variables:
threshold = 0.01  # using variance as a proxy for information in the column.

# Driver:
if __name__ == '__main__':

    # import the dataset for feature selection:
    X = pd.read_csv(Path('../..', '..', 'data', 'online_sales_dataset_for_fs.csv'), index_col=0)

    print(f"Number of features incoming: {X.shape[1]}")

    # initialize the selector:
    selector = VarianceThreshold(threshold=0.01)
    # fit
    selector.fit(X)
    # leave selected features in the original dataframe:
    X = X[X.columns[selector.get_support(indices=True)]]

    print(f"Number of features outgoing: {X.shape[1]}")

    # restore the feature names:


    # save the dataset:
    pd.DataFrame(X).to_csv(Path('..', '..', '..', 'data', 'online_sales_dataset_fs_variance_threshold.csv'))

    # We removed features which would likely not add information to the model.

