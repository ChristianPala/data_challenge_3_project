# Auxiliary libray to load the Online Sales dataset:
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd

# Timing:
from auxiliary.method_timer import measure_time


# Functions:
@measure_time
def load_and_save_data() -> pd.DataFrame:
    """
    Load the two datasets and save the combined dataset.
    :return: The combined dataset as a pandas dataframe.
    """
    # We created 2 csv files from the Excel sheet provided by professor Mitrovic to speed up the loading process:
    try:
        df_1_path: Path = Path('..', 'data', 'online_sales_2009_2010_dataset.csv')
        df_2_path: Path = Path('..', 'data', 'online_sales_2010_2011_dataset.csv')
        # Load data
        df_09 = pd.read_csv(df_1_path, sep=';')
        df_10 = pd.read_csv(df_2_path, sep=';')
        # merge the two datasets for preprocessing, and save to a file:
        pd.concat([df_09, df_10], axis=0).to_csv(Path('..', 'data', 'online_sales_dataset.csv'), index=False)
        df = pd.read_csv(Path('..', 'data', 'online_sales_dataset.csv'))

    # If they are not available, load the original Excel sheet:
    except FileNotFoundError:
        dict_df = pd.read_excel(Path('..', 'data', 'online_sales_dataset.xlsx'),
                                sheet_name=['Year 2009-2010', 'Year 2010-2011'])

        df_09 = dict_df.get('Year 2009-2010')
        df_10 = dict_df.get('Year 2010-2011')
        # merge the two datasets for preprocessing, and save to a file:
        pd.concat([df_09, df_10], axis=0).to_csv(Path('..', 'data', 'online_sales_dataset.csv'), index=False)
        df = pd.read_csv(Path('..', 'data', 'online_sales_dataset.csv'))

    # fix the date, works for both windows and ubuntu:
    try:
        # windows:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M')
    except ValueError:
        # ubuntu:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y-%m-%d %H:%M')

    return df
