import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fs_forward_selection.csv'))
    print(df)

    sns.heatmap(df.corr('pearson'))
    plt.show()
