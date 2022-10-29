# use T-sne to reduce the dimensions of the dataset, both as a visualization tool
# and to check how the model performs with a smaller number of features
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Dimensionality reduction:
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Driver:
if __name__ == '__main__':
    # load the dataset for dimensionality reduction:
    X = pd.read_csv(Path('..', 'data', 'online_sales_dataset_for_dr.csv'), index_col=0)

    # scale the data with the standard scaler:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # initialize the TSNE, since PCA is not performing well, we don't use it as an initializer
    tsne = TSNE(n_components=2, verbose=1, perplexity=40,  learning_rate='auto')
    # fit
    tsne_results = tsne.fit_transform(X_scaled)

    # save the results to the dataframe
    X['tsne_2_f1'] = tsne_results[:, 0]
    X['tsne_2_f2'] = tsne_results[:, 1]

    # remove all the features except the two we just created:
    X = X[['tsne_2_f1', 'tsne_2_f2']]

    # save the dataframe as a csv for model evaluation and visualization:
    X.to_csv(Path('..', 'data', 'online_sales_dataset_dr_tsne.csv'))
