# use t-SNE to reduce the dimensions of the dataset, both as a visualization tool
# and to check how the model performs with a smaller number of features
# Libraries:
# Data manipulation:

import pandas as pd
from pathlib import Path

# Dimensionality reduction:
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Global variables:
# Dimension for the t-SNE:
nr_of_components: int = 3
# Initialization for the t-SNE:
initialization = 'pca'
perplexity: int = 40

# Driver:
if __name__ == '__main__':
    # load the dataset for dimensionality reduction:
    X = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_dr.csv'))

    # save the CustomerId column for later use, drop it from the dataset:
    customer_id = X['CustomerId']
    X.drop('CustomerId', axis=1, inplace=True)

    # initialize the TSNE, since PCA is performing well. and it's advised in the documentation, we will
    # use the PCA initialization:
    tsne = TSNE(n_components=nr_of_components, init=initialization, verbose=1, perplexity=perplexity,
                learning_rate='auto')
    # tested perplexity based on this article: https://distill.pub/2016/misread-tsne/
    # tried 5, 30, 40 and 50 and N^0.5, where N is the number of samples, 40 gave the best visual results

    # fit
    tsne_results = tsne.fit_transform(X)

    # save the results as a dataframe, add the CustomerID as the first column:
    tsne_results_df = \
        pd.DataFrame(tsne_results, columns=[f'tsne_{nr_of_components}_f{i}'
                                            for i in range(1, nr_of_components + 1)])
    tsne_results_df.insert(0, 'CustomerId', customer_id)

    if initialization == "warn":
        # save the results:
        tsne_results_df.to_csv(Path('..', '..', 'data', f'online_sales_dataset_dr_tsne.csv'), index=False)

    else:
        # save the results:
        tsne_results_df.to_csv(Path('..', '..', 'data', f'online_sales_dataset_dr_tsne_{initialization}.csv'),
                               index=False)
