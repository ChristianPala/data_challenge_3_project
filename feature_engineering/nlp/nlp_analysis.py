# analysis on the clusters generate with the nlp approach:
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


def main():
    similarity_threshold = 0.8

    # read the dataset:
    df = pd.read_csv(Path('data', f'online_sales_dataset_clusters_{similarity_threshold}.csv'))

    # print average purity:
    print(f'Average purity: {df["ClusterPurity"].mean()}')

    # print average number of customers per cluster:
    print(f'Average number of customers per cluster: {df["ClusterSize"].mean()}')

    # plot the distribution of the cluster sizes:
    plt.hist(df['ClusterSize'])
    plt.title('Distribution of the cluster sizes')
    plt.xlabel('Cluster size')
    plt.ylabel('Customer count')
    plt.show()

    # unfortunately either the clusters are too small or the purity is very low, it's unlikely
    # we will be able to use the clustering to improve the model, we leave it in the list of features
    # for the feature selection step in case it's useful for some marginal improvement.


if __name__ == '__main__':
    main()
