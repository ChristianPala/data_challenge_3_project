# Auxiliary library to combine the results from the wrapper methods
# based on feature commonality and performance with the tuned XGB model
# with embedded methods based on feature importance.

# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Driver:
if __name__ == '__main__':
    # load the dataset for feature selection:
    X = pd.read_csv(Path('..', 'data', 'online_sales_dataset_for_fs.csv'))

    # In this version we used the result of the forward selection with cross-validation and automatic
    # number of features selected:
    best_features = \
        """
CustomerId, index
Recency,0.4483233
1_Area under the curve,0.37265098
0_FFT mean coefficient_4,0.05259183
62,0.05008799
85,0.04118452
72,0.03516138
"""
    # select the features above:
    features = [x.split(',')[0] for x in best_features.split('\n') if x]

    # keep only the features above in X:
    X = X[features]

    # rename the columns:
    X.rename(columns={'CustomerId': 'CustomerID',
                      'Recency': 'Recency',
                      '1_Area under the curve': 'AverageDaysBetweenPurchaseAUC',
                      '0_FFT mean coefficient_4': 'TotalSpentFFTMeanCoefficient4',
                      '62': 'CustomerGraphDeepwalkEmbedding62of128',
                      '85': 'CustomerGraphDeepwalkEmbedding85of128',
                      '72': 'CustomerGraphDeepwalkEmbedding72of128'}, inplace=True)

    # save the dataset:
    X.to_csv(Path('..', 'data', 'online_sales_dataset_for_dr_fs_and_mi_0.1.csv'), index=False)
