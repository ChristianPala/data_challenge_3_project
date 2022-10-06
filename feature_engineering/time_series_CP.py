# Libraries:
import pandas as pd
from tsfresh import extract_features, select_features
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


if __name__ == '__main__':
    # read the dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_cleaned.csv'), index_col='CustomerId',
                     parse_dates=['InvoiceDate'])

    # create a new column with the month of the purchase:
    df['Month'] = df['InvoiceDate'].dt.to_period('M')

    # create a new colum with the day of the week of the purchase:
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    
    # plot the number of purchases per month:
    df.groupby('Month')['Invoice'].count().plot()
    plt.title('Number of purchases per month')
    plt.xlabel('Month')
    plt.ylabel('Number of purchases')
    plt.show()

    # plot the number of purchases per day of the week:
    df.groupby('DayOfWeek')['Invoice'].count().plot()
    plt.title('Number of purchases per day of the week')
    plt.xlabel('Day of the week')
    plt.ylabel('Number of purchases')
    plt.show()