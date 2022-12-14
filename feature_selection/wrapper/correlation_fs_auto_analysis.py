import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load the dataset with the features selected by forward selection:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_dr.csv'), index_col=0)

    # shorten the column names for readability:
    df.columns = [col.replace('DeepwalkEmbedding', 'DWEmb') for col in df.columns]
    df.columns = [col.replace('AverageDaysBetweenPurchase', 'AvgDaysBtwBuy') for col in df.columns]
    df.columns = [col.replace('TotalSpent', 'TotSpent') for col in df.columns]
    df.columns = [col.replace('CustomerGraph', 'CIdGrph') for col in df.columns]
    df.columns = [col.replace('MeanCoefficient', 'MeanCoeff') for col in df.columns]

    # correlation matrix:
    corr = df.corr('spearman').round(2)
    # Note: pearson looks for linear relationships, spearman looks for monotonic relationships,
    # including non-linear ones.
    # mask the upper triangle:
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr[mask] = np.nan
    # get the column values:
    y_tick = corr.columns.values
    # replace the first y_tick:
    y_tick = [col.replace('AvgOrderDays_Area under the curve', '') for col in y_tick]

    # plot:
    plt.figure(figsize=(20, 20))
    # decrease the font size:
    sns.set(font_scale=1.2)
    sns.heatmap(corr, annot=True, cmap='coolwarm',
                xticklabels=corr.columns.values[:-1],
                yticklabels=y_tick,
                linewidths=0.2, vmin=-1, vmax=1)
    plt.title('Correlation Matrix with Spearman Correlation')
    plt.xticks(rotation=90)
    # give more space to the y-axis labels:
    plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.2)

    plt.show()

    # The correlation matrix shows how there is some correlation between the features, but not
    # a lot.

    # save the correlation matrix:
    corr.to_csv(Path('..', '..', 'plots', 'automated_forward_selection_correlation_matrix.csv'))
