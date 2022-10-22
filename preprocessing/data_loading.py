# Libraries:
from pathlib import Path
import pandas as pd


# Functions:
def load_and_save_data() -> pd.DataFrame:
    """
    Load the two datasets and save the combined dataset.
    @param df_1_path: Path to the first dataset from the Excel sheet provided by professor Mitrovic.
    @param df_2_path: Path to the second dataset from the Excel sheet provided by professor Mitrovic.
    :return: The combined dataset as a pandas dataframe.
    """
    try:

        df_1_path: Path = Path('..', 'data', 'online_sales_2009_2010_dataset.csv')
        df_2_path: Path = Path('..', 'data', 'online_sales_2010_2011_dataset.csv')
        # Load data
        df_09 = pd.read_csv(df_1_path, sep=';')
        df_10 = pd.read_csv(df_2_path, sep=';')
        # merge the two datasets for preprocessing, and save to a file:
        pd.concat([df_09, df_10], axis=0).to_csv(Path('..', 'data', 'online_sales_dataset.csv'), index=False)
        df = pd.read_csv(Path('..', 'data', 'online_sales_dataset.csv'))

    except FileNotFoundError:
        dict_df = pd.read_excel(Path('..', 'data', 'online_sales_dataset.xlsx'), sheet_name=['Year 2009-2010', 'Year 2010-2011'])

        df_09 = dict_df.get('Year 2009-2010')
        df_10 = dict_df.get('Year 2010-2011')
        # merge the two datasets for preprocessing, and save to a file:
        pd.concat([df_09, df_10], axis=0).to_csv(Path('..', 'data', 'online_sales_dataset.csv'), index=False)
        df = pd.read_csv(Path('..', 'data', 'online_sales_dataset.csv'))

        # fix the date
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y/%m/%d %H:%M:%S')

    return df
