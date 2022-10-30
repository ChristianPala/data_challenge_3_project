# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from modelling.feature_selection_performances.evaluator import evaluate_csv

# Driver:
if __name__ == '__main__':
    # load the paths:
    baseline = Path('..', '..', 'data', 'online_sales_dataset_for_dr.csv')
    pca_fs = Path('..', '..', 'data', 'online_sales_dataset_dr_pca.csv')
    tsne_fs = Path('..', '..', 'data', 'online_sales_dataset_dr_tsne.csv')
    pca_full = Path('..', '..', 'data', 'online_sales_dataset_dr_pca_full.csv')
    tsne_full = Path('..', '..', 'data', 'online_sales_dataset_dr_tsne_full.csv')

    # evaluate the PCA fs dataset:
    evaluate_csv(pca_fs, 'pca', fast=True)
    # evaluate the TSNE fs dataset:
    evaluate_csv(tsne_fs, 't-SNE', fast=True)
    # evaluate the PCA dataset with all the features:
    evaluate_csv(pca_full, 'pca_full', fast=True)
    # evaluate the TSNE dataset with all the features:
    evaluate_csv(tsne_full, 't-SNE_full', fast=True)
