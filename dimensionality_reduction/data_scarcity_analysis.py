# Auxiliary library to check how scarce our dataset is,
# important to determine the more appropriate dimensionality
# reduction methods
# Libraries:
# Data Manipulation:
import pandas as pd
from pathlib import Path

# Driver:
if __name__ == '__main__':
    # import the dataset for dimensionality reduction:
    # Todo: change to the dataset with the features selected by the wrapper method once it is ready.
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_for_fs_mutual_information.csv'),
                     index_col=0)

    # convert the dataframe to a SparseArray
    spar_df = df.apply(pd.arrays.SparseArray)

    print(f"The sparsity density is: {spar_df.sparse.density:.3f}")
    print(f"The dataset is basically full, PCA should work well")
