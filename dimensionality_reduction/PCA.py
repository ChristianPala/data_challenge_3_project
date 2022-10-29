# perform PCA on the dataset with the features selected by the wrapper method:
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Dimensionality reduction:
from sklearn.decomposition import PCA

# Scaling:
from sklearn.preprocessing import StandardScaler

# Driver:
if __name__ == '__main__':
    # import the dataset for dimensionality reduction:
    X = pd.read_csv(Path('..', 'data', 'online_sales_dataset_for_dr.csv'), index_col=0)

    # get the number of features:
    n_features = X.shape[1]

    # Scale the data before performing PCA:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for nr_components in range(n_features):
        # initialize the selector:
        pca = PCA(n_components=nr_components)
        # fit
        principal_components = pca.fit_transform(X_scaled)

        # cast the principal components to a dataframe:
        principal_components_df = pd.DataFrame(principal_components,
                                               columns=[f"PC{i}" for i in range(1, nr_components + 1)])

        # print the sum of the explained variance ratio:
        print(f"Sum of the explained variance ratio for {nr_components} components: "
              f"{pca.explained_variance_ratio_.sum():.3f}")

    # based on the analysis, we will keep 9 components using .8 as the threshold:
    # initialize the selector:
    pca = PCA(n_components=9)
    # fit
    principal_components = pca.fit_transform(X_scaled)

    # cast the principal components to a dataframe:
    principal_components_df = pd.DataFrame(principal_components,
                                           columns=[f"PC{i}" for i in range(1, 10)])

    # save the principal components:
    principal_components_df.to_csv(Path('..', 'data', 'online_sales_dataset_dr_pca.csv'))
