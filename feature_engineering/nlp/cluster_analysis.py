# analysis on the clusters generate with the nlp approach:
import pandas as pd
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

if __name__ == '__main__':

    # read the dataset:
    df_c = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_clusters_0.8.csv'))

    # print average purity:
    print(f'Average purity: {df_c["ClusterPurity"].mean()}')

    # print average number of customers per cluster:
    print(f'Average number of customers per cluster: {df_c["NumberOfCustomers"].mean()}')

    # plot the distribution of the cluster sizes:
    plt.hist(df_c['ClusterSize'])
    plt.title('Distribution of the cluster sizes')
    plt.xlabel('Cluster size')
    plt.ylabel('Customer count')
    plt.show()