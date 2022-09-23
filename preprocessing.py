# Libraries:
import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == '__main__':
    # data filepath
    data_path = Path('data/online_sales_dataset.xlsx')
    # Load data
    data = pd.read_csv(data_path, sep = ';')
    # First look
    print(data.head())
