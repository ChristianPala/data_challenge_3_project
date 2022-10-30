# use T-sne to reduce the dimensions of the dataset, both as a visualization tool
# and to check how the model performs with a smaller number of features
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Dimensionality reduction:
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Global variables:
nr_of_components: int = 3

# Driver:
if __name__ == '__main__':
    # load the dataset for dimensionality reduction:
    X = pd.read_csv(Path('../..', 'data', 'online_sales_dataset_for_dr.csv'))

    # save the CustomerID column for later use, drop it from the dataset:
    customer_id = X['CustomerID']
    X.drop('CustomerID', axis=1, inplace=True)

    # scale the data with the standard scaler:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # initialize the TSNE, since PCA is not performing well, we don't use it as an initializer
    tsne = TSNE(n_components=nr_of_components, verbose=1, perplexity=40,  learning_rate='auto')
    # fit
    tsne_results = tsne.fit_transform(X_scaled)

    # save the results as a dataframe, add the CustomerID as the first column:
    tsne_results_df = \
        pd.DataFrame(tsne_results, columns=[f'tsne_{nr_of_components}_f{i}'
                                            for i in range(1, nr_of_components + 1)])
    tsne_results_df.insert(0, 'CustomerID', customer_id)

    # save the results:
    tsne_results_df.to_csv(Path('../..', 'data', 'online_sales_dataset_dr_tsne.csv'), index=False)