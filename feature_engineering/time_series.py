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

    cfg = tsfel.get_features_by_domain()
    # print(len(cfg))
    # print(df.shape)

    # Extract features
    # running this will give you a warning on line 300 of calc_features.py, you might want to change
    # the line with this: features_final = pd.concat([features_final, feat]), or just comment the original and use this
    X = tsfel.time_series_features_extractor(cfg, df, verbose=0, window_size=15)  # how to choose window size????

    print(X.shape)  # window_size=100-> 58, 1110) window_size=15->(388, 852) now tell me how tf we gonna decide on ws..

    # some plots to explain stuff
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
