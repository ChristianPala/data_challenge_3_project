# Libraries:
import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == '__main__':
    # data filepath
    data_path = Path('data/online_sales_2009_2010_dataset.csv')
    # Load data
    data = pd.read_csv(data_path, sep=';')
    # First look
    print(data.head())
