# Libraries:
from pathlib import Path
import pandas as pd


# Functions:
def load_and_save_data(df_1_path: Path = Path('..', 'data', 'online_sales_2009_2010_dataset.csv'),
                       df_2_path: Path = Path('..', 'data', 'online_sales_2010_2011_dataset.csv')) -> pd.DataFrame:

    # Load data
    df_09 = pd.read_csv(df_1_path, sep=';')
    df_10 = pd.read_csv(df_2_path, sep=';')
    # merge the two datasets for preprocessing, and save to a file:
    pd.concat([df_09, df_10], axis=0).to_csv(Path('..', 'data', 'online_sales_dataset.csv'), index=False)
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset.csv'))
    return df
