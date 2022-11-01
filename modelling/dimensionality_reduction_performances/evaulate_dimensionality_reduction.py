# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from modelling.feature_selection_performances.evaluate_features_selection import evaluate_csv

# Driver:
if __name__ == '__main__':
    # load the paths:
    baseline = Path('..', '..', 'data', 'online_sales_dataset_for_dr_fs.csv')
    pca_fs = Path('..', '..', 'data', 'online_sales_dataset_dr_pca.csv')
    tsne_fs = Path('..', '..', 'data', 'online_sales_dataset_dr_tsne.csv')
    tsne_fs_pca = Path('..', '..', 'data', 'online_sales_dataset_dr_tsne_pca.csv')

    # We checked and the results are much worse if we don't feature select before dimensionality reduction.
    # pca_full = Path('..', '..', 'data', 'online_sales_dataset_dr_pca_full.csv')
    # tsne_full = Path('..', '..', 'data', 'online_sales_dataset_dr_tsne_full.csv')

    # evaluate the baseline:
    evaluate_csv(baseline, 'baseline_fs_auto', fast=True)
    # evaluate the PCA fs dataset:
    evaluate_csv(pca_fs, 'pca', fast=True)
    # # evaluate the TSNE fs dataset:
    evaluate_csv(tsne_fs, 't-SNE', fast=True)
    evaluate_csv(tsne_fs_pca, 't-SNE + PCA', fast=True)

    # # evaluate the PCA full dataset:
    # # evaluate the PCA dataset with all the features:
    # evaluate_csv(pca_full, 'pca_full', fast=True)
    # # evaluate the TSNE dataset with all the features:
    # evaluate_csv(tsne_full, 't-SNE_full', fast=True)
    # Dimensionality reduction works better once we feature select the dataset.
