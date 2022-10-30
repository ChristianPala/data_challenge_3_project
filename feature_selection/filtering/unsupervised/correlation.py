import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv(Path('..', '..', '..', 'data', 'online_sales_dataset_for_dr_fs.csv'), index_col=0)

    # shorten the column names:
    df.columns = [col.replace('DeepwalkEmbedding', 'DWEmb') for col in df.columns]
    df.columns = [col.replace('AverageDaysBetweenPurchase', 'AvgDaysBtwBuy') for col in df.columns]
    df.columns = [col.replace('TotalSpent', 'TotSpent') for col in df.columns]

    plt.figure(figsize=(20, 15))
    # tight layout
    sns.heatmap(df.corr('pearson'), annot=True, cmap='RdYlGn', linewidths=0.2)
    # increase the font size
    plt.rcParams.update({'font.size': 18})
    plt.xticks(rotation=15)
    plt.show()

    # The correlation matrix confirms the features selected are not correlated very correlated with each other.

    # save the correlation matrix:
    df.corr('pearson').to_csv(Path('..', '..', '..', 'plots', 'automated_forwad_selection_correlation_matrix.csv'))