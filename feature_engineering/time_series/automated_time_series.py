# Libraries:

import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    # import the fe dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))


