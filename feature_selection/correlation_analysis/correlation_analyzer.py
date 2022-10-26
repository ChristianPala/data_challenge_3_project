# Library to perform correlation analysis between the engineered features and the target variable.
# Data manipulation:
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# Plotting:
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Driver:
if __name__ == '__main__':

    # import the RFM dataset:
    # Todo use the ful dataset when we have all the features
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    # get the features and the target:
    y = df['CustomerChurned']
    X = df.drop(['CustomerChurned', 'CustomerId', 'Description'], axis=1)

    # get the correlation between the features and the target:
    corr = X.corrwith(y)
    # sort the correlations:
    corr = abs(corr).sort_values(ascending=False)
    # display the correlations:
    print(f'Correlation magnitude with the target label:\n{round(100 * corr, 2)}%')

    # get the correlation between the features:
    corr = X.corr()
    # plot the correlation matrix:
    fig = plt.figure(figsize=(14, 14))
    plt.matshow(corr, fignum=fig.number, cmap="Blues")
    plt.xticks(range(len(corr.columns)), corr.columns, fontsize=10)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=10)
    c = plt.colorbar()
    c.ax.tick_params(labelsize=12)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

    # save the correlation matrix:
    # if the path does not exist, create it:
    if not Path('..', '..', 'plots', 'Feature Selection', 'Correlations').exists():
        Path('..', '..', 'plots', 'Feature Selection', 'Correlations').mkdir(parents=True)
    plt.savefig(Path('..', '..', 'plots', 'Feature Selection', 'Correlations', 'correlation_matrix.png'))

    # print the correlation matrix:
    print(f'Correlation matrix:\n{tabulate(corr, headers="keys", tablefmt="psql")}')

    # Total spent and total quantity are highly correlated, it's reasonable to keep only one of them.



