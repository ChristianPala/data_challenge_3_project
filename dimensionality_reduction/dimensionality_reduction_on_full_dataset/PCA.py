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
Threshold: float = 0.8

# Driver:
if __name__ == '__main__':
    # import the dataset for dimensionality reduction:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fs.csv'), index_col=0)

    # get the number of features:
    n_features = X.shape[1]

    # Scale the data before performing PCA:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_features)
    # fit
    principal_components = pca.fit_transform(X_scaled)

    # from 1 to the feature number, get the explained variance ratio:
    explained_variance_ratio = pca.explained_variance_ratio_
    # get the number of components needed to explain more than 80% of the variance:
    n_components = next(i for i, v in enumerate(explained_variance_ratio.cumsum(), 1) if v > Threshold)
    print(f"Threshold {Threshold} required {n_components} components "
          f"to explain {explained_variance_ratio.cumsum()[n_components - 1]:.2%} of the variance.")

    # print the explained variance ratio for each component:
    print(f"Explained variance ratio for each component: {explained_variance_ratio}")

    # keep 112 components to explain more than .8 of the variance:
    # initialize the selector:
    pca = PCA(n_components=n_components)
    # fit
    principal_components_final = pca.fit_transform(X_scaled)

    # cast the principal components to a dataframe:
    principal_components_df = pd.DataFrame(principal_components_final,
                                           columns=[f"PC{i}" for i in range(1, n_components + 1)])

    # Add the CustomerID column as the first column:
    principal_components_df.insert(0, 'CustomerID', X.index)

    # save the principal components:
    principal_components_df.to_csv(Path('..', '..', 'data', 'online_sales_dataset_dr_pca_full.csv'), index=False)
