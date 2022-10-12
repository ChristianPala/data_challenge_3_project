# libraries
from pathlib import Path
import pandas as pd
import matplotlib
import tsfel
matplotlib.use('tkagg')

if __name__ == '__main__':

    # import the aggregated dataset:
    df = pd.read_csv(Path('../..', 'data', 'online_sales_dataset_cleaned.csv'))

    # create the aggregated costumer dataset:
    df_agg = df.groupby('InvoiceDate').agg({'Invoice': 'count', 'Quantity': 'sum', 'Price': 'sum', 'Country': lambda x: x.value_counts().index[0]})
    df_agg.rename(columns={'Invoice': 'NumberOfPurchases', 'Quantity': 'TotalQuantity', 'Price': 'TotalSpent'},
                  inplace=True)
    df_agg.index = pd.to_datetime(df_agg.index)

    x = df_agg.index
    df_agg = df_agg[:100]

    # df = df[:10000]
    # remove the description column:
    # df.drop('Description', axis=1, inplace=True)
    # convert the invoice date column to datetime:

    # df_agg['InvoiceDate'] = pd.to_datetime(df_agg['InvoiceDate'])
    # df_agg['InvoiceDate'] = (df_agg['InvoiceDate'] - df_agg['InvoiceDate'].min()) / np.timedelta64(1, 'D')

    # Retrieves a pre-defined feature configuration file to extract all available features
    print(df_agg.shape[0])

    X_toreturn = pd.DataFrame()
    for i in range(df_agg.shape[0]):
        cfg = tsfel.get_features_by_domain()

        # Extract features
        X = tsfel.time_series_features_extractor(cfg, df_agg, verbose=0)
        X_toreturn = pd.concat([X_toreturn, X])

    # doing it on a small set, so it's faster...
    X_toreturn.to_csv(Path('../..', 'data', 'online_sales_dataset_ts_fe.csv'), index=False)
    # do not understand why the f it doesnt save the headers as column names, is it because
    # they start with numbers?? could be...but idk


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


