# libraries
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import tsfel
matplotlib.use('tkagg')

if __name__ == '__main__':

    # import the aggregated dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))
    # remove the description column:
    df.drop('Description', axis=1, inplace=True)
    # convert the invoice date column to datetime:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceDate'] = (df['InvoiceDate'] - df['InvoiceDate'].min()) / np.timedelta64(1, 'D')

    # Retrieves a pre-defined feature configuration file to extract all available features
    print(df.shape)

    cfg = tsfel.get_features_by_domain()

    # Extract features
    X = tsfel.time_series_features_extractor(cfg, df)
    # doing it on a small set, so it's faster...
    # not sure if we can use this stuff...
    X.to_csv(Path('..', 'data', 'online_sales_dataset_ts_fe.csv'), index=False)

    # # create the aggregated costumer dataset:
    # df_agg = df.groupby('InvoiceDate').agg({'Invoice': 'count', 'Quantity': 'sum', 'Price': 'sum', 'Country': lambda x: x.value_counts().index[0]})
    # df_agg.rename(columns={'Invoice': 'NumberOfPurchases', 'Quantity': 'TotalQuantity', 'Price': 'TotalSpent'},
    #               inplace=True)
    # df_agg.index = pd.to_datetime(df_agg.index)

    # # show the plots for number of purchases and total money spent
    # x = df_agg.index
    #
    # fig, ax = plt.subplots(2, 1)
    # ax[0].scatter(x=x, y=df_agg['NumberOfPurchases'], marker='.')
    # mean_n_purch = np.mean(df_agg['NumberOfPurchases']).astype(int)
    # npurch = df_agg['NumberOfPurchases']
    # y_average1 = npurch.rolling(window=mean_n_purch).mean()
    # ax[0].plot(x, y_average1, label='Rolling mean', linestyle='-', c='orange')  # mean line
    # ax[0].set_xlabel('datetime')
    # ax[0].set_ylabel('number of purchases')
    # ax[0].legend()
    #
    # ax[1].scatter(x, df_agg['TotalSpent'], marker='.')
    # mean_tot_spend = np.mean(df_agg['TotalSpent']).astype(int)
    # tot_spend = df_agg['TotalSpent']
    # y_average2 = tot_spend.rolling(window=mean_tot_spend).mean()
    # ax[1].plot(x, y_average2, label='Rolling mean', linestyle='-', c='orange')  # mean line
    # ax[1].set_xlabel('datetime')
    # ax[1].set_ylabel('money spent')
    #
    # ax[0].grid(True)
    # ax[1].grid(True)
    # plt.show()
    #
    # # we can clearly see that during the Xmas holidays purchases count increases,
    # # and so the total money spent (obviously)
    # #


