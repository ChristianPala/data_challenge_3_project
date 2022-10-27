# perform PCA on the dataset with the features selected by the wrapper method:
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Dimensionality reduction:
from sklearn.decomposition import PCA

# Scaling:
from sklearn.preprocessing import StandardScaler

# Global variables:
n_components = 5
# number of components to keep, do an analysis on how much variance is explained by each component to decide
# on the number of components to keep.


# Driver:
if __name__ == '__main__':
    # import the dataset for feature selection:
    # Todo: change to the dataset with the features selected by the wrapper method once it is ready.
    X = pd.read_csv(Path('../..', '..', 'data', 'online_sales_dataset_for_fs.csv'), index_col=0)

    # Scale the data before performing PCA:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # initialize the selector:
    pca = PCA(n_components=n_components)
    # fit
    principal_components = pca.fit_transform(X_scaled)

    # cast the principal components to a dataframe:
    principal_components = pd.DataFrame(principal_components, columns=[f"PC{i}" for i in range(1, n_components + 1)])

    # save the dataset:
    principal_components.to_csv(Path('..', 'data', 'online_sales_dataset_dim_reduction_pca.csv'))





