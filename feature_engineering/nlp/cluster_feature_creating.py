# Libraries:
import pandas as pd
from pathlib import Path

if __name__ == '__main__':
    # read the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))
    # read the clustered dataset:
    df_cluster = pd.read_csv(Path('..', 'data', 'online_sales_dataset_clusters_0.8.csv'))