# libraries
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import tsfel
# matplotlib.use('tkagg')

if __name__ == '__main__':

    # import the timeseries dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_timeseries.csv'))

    print(df.isna().sum())
    df.dropna(subset=['StockCode'], inplace=True)  # non ho proprio capito come mai non me le vuole droppare nel

    df = df.loc[:, ['CustomerId', 'InvoiceDate']]
    # timeseries_dataset.py, lì non me le trova nemmeno

    X = pd.DataFrame()

    cfg = tsfel.get_features_by_domain(json_path='features_mod.json')  # modified the json so that it doesnt calculate
    # LPCC (cause it gives errors due to too few rows in certain dataframes

    # Extract features
    # running this will give you a warning on line 300 of calc_features.py, you might want to change
    # the line with this: features_final = pd.concat([features_final, feat]), or just comment the original and use this

    # perform feature extraction on slices of the dataframe for every customer id,
    # saving all the data for that customer (invoice date, invoice code, etc.)
    # sort the dataframe
    df.sort_values(by='CustomerId', axis=0, inplace=True)
    # set the index to be this and don't drop
    df.set_index(keys=['CustomerId'], drop=False, inplace=True)
    # get a list of customers
    customers = df['CustomerId'].unique().tolist()

    print(customers)

    # now we can perform a lookup on a 'view' of the dataframe
    for customer in customers:
        # customers' dataframes
        this_personDF = df.loc[df.CustomerId == customer]
        # print(this_personDF.shape)
        # print(customers.index(customer))

        if this_personDF.shape[0] < 2:
            continue
        # returns ONE row for each the customer
        features_data = tsfel.time_series_features_extractor(cfg, this_personDF, verbose=0)
        X = pd.concat([X, features_data])

    print(X.shape)  # shape varies much based on the window size
    X.to_csv(Path('..', '..', 'data', 'online_sales_dataset_tsfel.csv'), index=False)

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
