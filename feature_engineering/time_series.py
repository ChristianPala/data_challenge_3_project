# libraries
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg') # just comment it if it works for you in pycharm, dunno why doesnt want to show in the sciview like b4

if __name__ == '__main__':

    # import the cleaned dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_cleaned.csv'))

    # create the aggregated costumer dataset:
    df_agg = df.groupby('InvoiceDate').agg({'Invoice': 'count', 'Quantity': 'sum', 'Price': 'sum', 'Country': lambda x: x.value_counts().index[0]})
    df_agg.rename(columns={'Invoice': 'NumberOfPurchases', 'Quantity': 'TotalQuantity', 'Price': 'TotalSpent'},
                  inplace=True)
    df_agg.index = pd.to_datetime(df_agg.index)

    # show the plots for number of purchases and total money spent
    x = df_agg.index

    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(x=x, y=df_agg['NumberOfPurchases'], marker='.')
    mean_n_purch = np.mean(df_agg['NumberOfPurchases']).astype(int)
    npurch = df_agg['NumberOfPurchases']
    y_average1 = npurch.rolling(window=mean_n_purch).mean()
    ax[0].plot(x, y_average1, label='Rolling mean', linestyle='-', c='orange')  # mean line
    ax[0].set_xlabel('datetime')
    ax[0].set_ylabel('number of purchases')
    ax[0].legend()

    ax[1].scatter(x, df_agg['TotalSpent'], marker='.')
    mean_tot_spend = np.mean(df_agg['TotalSpent']).astype(int)
    tot_spend = df_agg['TotalSpent']
    y_average2 = tot_spend.rolling(window=mean_tot_spend).mean()
    ax[1].plot(x, y_average2, label='Rolling mean', linestyle='-', c='orange')  # mean line
    ax[1].set_xlabel('datetime')
    ax[1].set_ylabel('money spent')

    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()

    # we can clearly see that during the Xmas holidays purchases count increases,
    # and so the total money spent (obviously)
    #


