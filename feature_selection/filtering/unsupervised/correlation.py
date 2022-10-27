import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv(Path('../..', '..', 'data', 'FS_forward_timeseries.csv'))
    print(df)

    sns.heatmap(df.corr('pearson'))
    plt.show()
