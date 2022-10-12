# Libraries:

import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    df_ts = pd.read_csv(Path('../..', 'data', 'online_sales_dataset_agg'), index_col='id')