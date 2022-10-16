# analysis on the clusters generate with the nlp approach:
import pandas as pd
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

if __name__ == '__main__':

    similarity_threshold = 0.7

    # read the dataset:
    df = pd.read_csv(Path('..', '..', 'data', f'online_sales_dataset_clusters_{similarity_threshold}.csv'))

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

    # with 0.7 as a threshold we have an average purity slightly better compared to
    # the f-score of the RFM model (0.722 vs 0.7) so we may have better predictors
    # in the test set where we can use the clusters to predict the target variable.


