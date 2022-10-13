# analysis on the clusters generate with the nlp approach:
import pandas as pd
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

if __name__ == '__main__':

    # read the dataset:
    df_c = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_clusters_0.8.csv'))
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    # cast the ClusterId as a set of floats:
    df_c['ClusterId'] = df_c['ClusterId'].apply(lambda x: set([float(i) for i in x[0:-1].split(',')]))
    df_c['ClusterSize'] = df_c['ClusterId'].apply(lambda x: len(x))
    df_c.drop('CustomerId', axis=1, inplace=True)

    # Keep only the first row of each cluster id:
    df_c = df_c.drop_duplicates(subset='ClusterId', keep='first')

    # For each cluster, check the purity with the CustomerChurned column:
    df_c['Churned'] = df_c['ClusterId'].apply(lambda x: df[df['CustomerId'].isin(x)]['CustomerChurned'].sum())
    df_c['Churned'] = df_c['Churned'] / df_c['ClusterSize']
    # create a new column with the churned purity, 1 if all churned or did not churn, 0 otherwise:
    df_c['ClusterPurity'] = df_c['Churned'].apply(lambda x: 1 if x == 0 or x == 1 else 0)

    # print average purity:
    print(f'Average purity: {df_c["ClusterPurity"].mean()}')

    # plot the distribution of the cluster sizes:
    plt.hist(df_c['ClusterSize'])
    plt.title('Distribution of the cluster sizes')
    plt.xlabel('Cluster size')
    plt.ylabel('Customer count')
    plt.show()

    # save the dataset:
    df_c.to_csv(Path('..', '..', 'data', 'online_sales_dataset_clusters_0.8_purity.csv'), index=False)