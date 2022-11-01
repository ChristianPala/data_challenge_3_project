# Auxiliary library to visualize the t-SNE 3D plot
# Libraries:
# Data Manipulation:
from pathlib import Path
import pandas as pd

# Visualization:
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')


# Driver:
if __name__ == '__main__':
    # load the t-SNE dataset:
    t_sne = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_dr_tsne_full.csv'), index_col=0)

    # load the target:
    target = pd.read_csv(Path('..', '..', 'data', 'online_sales_labels_tsfel.csv'), index_col=0)

    # create a dataframe with the target and the t-SNE features:
    df_subset = pd.DataFrame()
    df_subset['tsne_3_f1'] = t_sne['tsne_3_f1']
    df_subset['tsne_3_f2'] = t_sne['tsne_3_f2']
    df_subset['tsne_3_f3'] = t_sne['tsne_3_f3']
    df_subset['y'] = target.CustomerChurned.astype(int)

    # create a figure:
    fig = plt.figure(figsize=(16, 10))

    # create a 3D scatter plot:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_subset.tsne_3_f1, df_subset.tsne_3_f2, df_subset.tsne_3_f3, c=df_subset.y,
               cmap=mpl.colormaps["viridis"], s=20, alpha=0.5, edgecolors='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('t-SNE 3D Plot of the Online Sales Dataset')
    ax.legend(['Churned', 'Not Churned'], loc='upper left')
    plt.show()

