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
    # PCA visualizations:
    pca = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_dr_pca.csv'), index_col=0)

    nr_comp = pca.shape[1]

    target = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # build the visualization dataframe:
    df_subset = pd.DataFrame()
    df_subset[f'pca-1'] = pca.iloc[:, 0]
    df_subset[f'pca-2'] = pca.iloc[:, 1]
    df_subset[f'pca-3'] = pca.iloc[:, 2]
    df_subset['y'] = target.CustomerChurned

    # 2D visualization:
    sns.scatterplot(
        x=f'pca-1', y=f'pca-2',
        # tested combinations, and 1 and 2 are the best:
        hue='y', hue_order=[1, 0],
        palette=sns.color_palette('hls', 2),
        data=df_subset,
        legend=False,
        alpha=0.8,
        s=20,
        edgecolors='k'
    )
    plt.legend(loc='upper left', labels=['Churner', 'Not churner'])
    plt.title('2D PCA visualization')

    try:
        plt.savefig(Path('..', '..', 'plots', 'PCA', 'pca_visualization2D.png'))
    except FileNotFoundError:
        Path('..', '..', 'plots', 'PCA').mkdir(parents=True, exist_ok=True)
        plt.savefig(Path('..', '..', 'plots', 'PCA', 'pca_visualization2D.png'))

    # 3D visualization:
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df_subset['pca-1'],
        ys=df_subset['pca-2'],
        zs=df_subset['pca-3'],
        c=df_subset['y'],
        cmap=plt.cm.rainbow
    )
    ax.set_xlabel(f'pca-1')
    ax.set_ylabel(f'pca-2')
    ax.set_zlabel(f'pca-3')
    ax.legend(['Churned', 'Not churned'])
    title = f'3D PCA visualization'
    plt.title(title)

    try:
        plt.savefig(Path('..', '..', 'plots', 'PCA', 'pca_visualization3D.png'))
    except FileNotFoundError:
        Path('..', '..', 'plots', 'PCA').mkdir(parents=True, exist_ok=True)
        plt.savefig(Path('..', '..', 'plots', 'PCA', 'pca_visualization3D.png'))
    plt.show()
