# use T-sne to reduce the dimensions of the dataset, both as a visualization tool
# and to check how the model performs with a smaller number of features
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Dimensionality reduction:
from sklearn.manifold import TSNE

# Driver:
if __name__ == '__main__':
    # load the dataset from the PCA analysis:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_dim_reduction_pca.csv'))

    # initialize the TSNE:
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # fit
    tsne_results = tsne.fit_transform(df)

    # save the results to the dataframe
    df['tsne_2_f1'] = tsne_results[:, 0]
    df['tsne_2_f2'] = tsne_results[:, 1]

    # save the dataframe as a csv for model evaluation and visualization:
    df.to_csv(Path('..', 'data', 'online_sales_dataset_dim_reduction_pca_tsne.csv'))
