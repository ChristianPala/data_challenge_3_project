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
    pca = Path('..', '..', 'data', 'online_sales_dataset_dr_pca.csv')
    tsne = Path('..', '..', 'data', 'online_sales_dataset_dr_tsne.csv')

    # evaluate the dataset with all the features:
    evaluate_csv(baseline, 'baseline_dr', fast=True)
    # evaluate the PCA dataset:
    evaluate_csv(pca, 'pca', fast=True)
    # evaluate the TSNE dataset:
    evaluate_csv(tsne, 't-SNE', fast=True)
