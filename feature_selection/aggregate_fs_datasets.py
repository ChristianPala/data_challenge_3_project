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

    # In this version we selected the features with importance > 0.005 using the embedded method.
    best_features = \
        """
CustomerId, index
1_Area under the curve,0.0549434
1_Spectral slope,0.0187276
1_Spectral variation,0.0132853
0_ECDF_1,0.0101371
ProductPagerank,0.00951809
0_ECDF_0,0.00933946
ProductEigenvectorCentrality,0.0085773
ProductBetweennessCentrality,0.00814724
ProductClosenessCentrality,0.00773604
20,0.00578936
0_ECDF_7,0.00569517
DegreeCentrality,0.00553207
EigenvectorCentrality,0.0050479
"""

    # select the features above:
    features = [x.split(',')[0] for x in best_features.split('\n') if x]

    # keep only the features above in X:
    X = X[features]

    X.columns = [x.replace('0_', 'TotSpent_') for x in X.columns]
    X.columns = [x.replace('1_', 'AvgOrderDays_') for x in X.columns]
    X.columns = [x.replace('20', 'CIDeepwalkEmb20-128') for x in X.columns]

    # save the dataset:
    X.to_csv(Path('..', 'data', 'online_sales_dataset_for_dr.csv'), index=False)
