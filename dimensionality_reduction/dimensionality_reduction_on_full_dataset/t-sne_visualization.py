# Auxiliary library to visualize the t-SNE 3D plot
# Libraries:
# Data Manipulation:
from pathlib import Path
import pandas as pd

# Visualization:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.use('tkagg')


# Driver:
if __name__ == '__main__':
    # load the t-SNE dataset:
    t_sne = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_dr_tsne_full.csv'), index_col=0)

    nr_of_features = len(t_sne.columns)

    # load the target:
    target = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # create a dataframe with the target and the t-SNE features:
    df_subset = pd.DataFrame()
    # add the t-SNE features:
    for i in range(1, nr_of_features + 1):
        df_subset['tsne_' + str(nr_of_features) + '_f' + str(i)] = t_sne['tsne_' + str(nr_of_features) + '_f' + str(i)]
    # add the target:
    df_subset['y'] = target.CustomerChurned.astype(int)

    if nr_of_features in (2, 3):
        # create a figure:
        fig = plt.figure(figsize=(16, 10))
        # create a 2D scatter plot:
        sns.scatterplot(x=f'tsne_{nr_of_features}_f1', y=f'tsne_{nr_of_features}_f2',
                        hue='y', data=df_subset, palette='coolwarm', s=20, alpha=0.8, edgecolors='k')
        plt.title('t-SNE 2D Plot of the Online Sales Dataset')
        plt.legend(['Churned', 'Not Churned'], loc='upper left')

        # save the figure:
        try:
            fig.savefig(Path('..', '..', 'plots', 't-SNE-PCA-init', 'online_sales_dataset_full_tsne_2D.png'))
        except FileNotFoundError:
            # create the directory:
            Path('..', '..', 'plots', 't-SNE-PCA-init').mkdir(parents=True, exist_ok=True)
            # save the figure:
            fig.savefig(Path('..', '..', 'plots', 't-SNE-PCA-init', 'online_sales_dataset_full_tsne_2D.png'))

    if nr_of_features == 3:
        # create a figure:
        fig = plt.figure(figsize=(16, 10))

        # create a 3D scatter plot:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_subset.tsne_3_f1, df_subset.tsne_3_f2, df_subset.tsne_3_f3, c=df_subset.y,
                   cmap=plt.cm.rainbow, s=20, alpha=0.5, edgecolors='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('t-SNE 3D Plot of the Online Sales Dataset')
        ax.legend(['Churned', 'Not Churned'], loc='upper left')
        plt.show()

        # save the figure:
        try:
            fig.savefig(Path('..', '..', 'plots', 't-SNE-PCA-init', 'online_sales_dataset_full_tsne_3D.png'))
        except FileNotFoundError:
            # create the directory:
            Path('..', '..', 'plots', 't-SNE-PCA-init').mkdir(parents=True, exist_ok=True)
            # save the figure:
            fig.savefig(Path('..', '..', 'plots', 't-SNE', 'online_sales_dataset_full_tsne_3D.png'))

    else:
        print('The t-SNE visualization is only available for 2D and 3D plots.')