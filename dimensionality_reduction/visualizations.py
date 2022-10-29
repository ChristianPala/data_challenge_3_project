# Data manipulation:
import matplotlib
import pandas as pd
import numpy as np
from pathlib import Path

# Data visualization
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

matplotlib.use('tkagg')


if __name__ == '__main__':
    # T-SNE viz
    t_sne = pd.read_csv(Path('..', 'data', 'online_sales_dataset_dr_tsne.csv'))
    pca = pd.read_csv(Path('..', 'data', 'online_sales_dataset_dr_pca.csv'))

    nr_comp = pca.shape[1] - 1  # -1 for the moment, I see that there is an Unnamed: 0 column

    target = pd.read_csv(Path('..', 'data', 'online_sales_labels_tsfel.csv'))

    df_subset = pd.DataFrame()
    df_subset['y'] = target.CustomerChurned

    df_subset[f'pca{nr_comp}-1'] = pca.iloc[:, 1]
    df_subset[f'pca{nr_comp}-2'] = pca.iloc[:, 2]
    df_subset[f'pca{nr_comp}-3'] = pca.iloc[:, 3]


    # ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    # ax.scatter(
    #     xs=df_subset['pca-1'],
    #     ys=df_subset['pca-2'],
    #     zs=df_subset['pca-3'],
    #     c=df_subset['y'],
    #     cmap='tab10'
    # )
    # ax.set_xlabel(f'pca{nr_comp}-1')
    # ax.set_ylabel(f'pca{nr_comp}-2')
    # ax.set_zlabel(f'pca{nr_comp}-3')
    # plt.show()

    df_subset['tsne_2_f1'] = t_sne.iloc[:, 1]
    df_subset['tsne_2_f2'] = t_sne.iloc[:, 2]

    ax1 = plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=f'pca{nr_comp}-1', y=f'pca{nr_comp}-2',
        hue='y',
        palette=sns.color_palette('hls', 2),
        data=df_subset,
        legend='full',
        ax=ax1
    )
    # print(df_subset)
    ax2 = plt.subplot(1, 2, 2)
    sns.scatterplot(
        x='tsne_2_f1', y='tsne_2_f2',
        hue='y',
        palette=sns.color_palette('hls', 2),
        data=df_subset,
        legend='full',
        ax=ax2
    )
    plt.show()


