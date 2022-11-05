# analyse recency and 1_AUC, the best features:
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from modelling.feature_selection_performances.evaluate_features_selection import evaluate_csv

# Driver:
if __name__ == '__main__':

    # load the feature selection dataset:
    fs_dataset = Path('../..', 'data', 'online_sales_dataset_for_dr_fs.csv')

    # keep only the recency and 1_AUC features:
    df = pd.read_csv(fs_dataset, index_col=0)
    df = df[['AverageDaysBetweenPurchaseAUC']]

    # save the dataset:
    df.to_csv(Path('../..', 'data', 'online_sales_dataset_vip_features.csv'))

    vip_path = Path('../..', 'data', 'online_sales_dataset_vip_features.csv')

    # evaluate the features:
    evaluate_csv(vip_path, 'AverageDaysBetweenPurchaseAUC')
