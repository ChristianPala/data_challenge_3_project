# libraries
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np
import tsfel
from tqdm import tqdm
import warnings
# suppress warnings
warnings.filterwarnings("ignore")


def feature_extractor(data, features, customer):
    # customers dataframes
    this_personDF = data.loc[data.CustomerId == customer]
    # returns ONE row for each the customer
    features_data = tsfel.time_series_features_extractor(features, this_personDF, verbose=0)
    return features_data


if __name__ == '__main__':

    # import the timeseries dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))

    df = df[['CustomerId', 'InvoiceDate']]  # cut df down to 2 columns

    # cast the invoice date to datetime, then to int and divide by 10^9:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y-%m-%d %H:%M:%S')
    df['InvoiceDate'] = df['InvoiceDate'].astype(np.int64) // 10 ** 9

    # skip the customers that occur only one time
    is_multi = df['CustomerId'].value_counts() > 1
    df = df[df['CustomerId'].isin(is_multi[is_multi].index)]

    cfg = tsfel.get_features_by_domain(json_path='features_mod.json')  # modified the json so that it doesnt calculate
    # LPCC (cause it gives errors due to too few rows in certain dataframes)

    # sort the dataframe
    df.sort_values(by='CustomerId', axis=0, inplace=True)
    # set the index to be this and don't drop
    df.set_index(keys=['CustomerId'], drop=False, inplace=True)
    # get a list of customers
    customers = df['CustomerId'].unique().tolist()

    print('> execution started')

    # execute the feature extraction in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(feature_extractor, df, cfg, customer) for customer in tqdm(customers)]
        # wait for all the futures to finish
        results = [future.result() for future in tqdm(futures)]
        # catch exceptions:
        for future in futures:
            if future.exception() is not None:
                print(future.exception())
                # remove the future from the list
                futures.remove(future)
                # print the customer id that caused the exception
                print(customers[futures.index(future)])
        # concatenate the results as a dataframe
        X = pd.concat(results)

    print('> task mapped')

    print(X.shape)
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
