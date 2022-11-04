# Auxiliary library to perform the exploratory data analysis on the online sales dataset.
from pathlib import Path

import numpy as np
import pandas as pd

# Libraries:
# Data manipulation:
from data_loading import load_and_save_data
# EDA:
import scipy
# Plotting:
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# matplotlib.use('tkagg')

# Driver:
if __name__ == '__main__':
    # load the dataset:
    df = load_and_save_data()

    # check the dataset:
    print(df.head())

    # check the dataset info:
    print(df.info())

    # check the dataset statistics:
    print(df.describe())

    # check the dataset statistics for columns saved as categorical:
    print(df.describe(include=['O']))

    # check the dataset statistics for columns saved as numerical:
    print(df.describe(include=['int64', 'float64', 'datetime64'], datetime_is_numeric=True))

    """
    There are a few problems:
    - We have negative values for quantity. We will solve it in the data cleaning phase.
    - Price is not saved as a float.
    """
    # cast the price column to float:
    df['Price'] = df['Price'].replace(',', '.').astype(float)

    # check the min and max values for the price column:
    print(f"Min price: {df['Price'].min()}")
    print(f"Max price: {df['Price'].max()}")

    """
    We also have negative values for price. We will solve it in the data cleaning phase.
    """

    # Numerical variables:
    # --------------------------------------------------------------

    # plot the correlation matrix, excluding the CustomerId column:
    plt.figure(figsize=(10, 10))
    corr_matrix = df.drop('Customer ID', axis=1).corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation matrix')
    plt.savefig(Path('EDA', 'correlation_matrix.png'))
    # as expected, the quantity and price are not correlated.
    plt.close()

    # plot the boxplot for the quantity column
    plt.figure(figsize=(10, 10))
    sns.boxplot(x=df['Quantity'], showfliers=True, showmeans=True, meanline=True)
    plt.title('Boxplot for the quantity column')
    plt.savefig(Path('EDA', 'boxplot_quantity.png'))
    plt.close()

    # plot the boxplot for the price column
    plt.figure(figsize=(10, 10))
    sns.boxplot(x=df['Price'], showfliers=True, showmeans=True, meanline=True)
    plt.title('Boxplot for the price column')
    plt.savefig(Path('EDA', 'boxplot_price.png'))
    plt.close()

    # distribution of the quantity column:
    plt.figure(figsize=(10, 10))
    sns.distplot(df['Quantity'], kde=False)
    plt.title('Distribution of the quantity column')
    plt.savefig(Path('EDA', 'distribution_quantity.png'))
    plt.close()

    # distribution of the price column:
    plt.figure(figsize=(10, 10))
    sns.distplot(df['Price'], kde=False)
    plt.title('Distribution of the price column')
    plt.savefig(Path('EDA', 'distribution_price.png'))
    plt.close()

    # it's too early to say much on the distributions with the problems noted above.

    # Categorical variables:
    # --------------------------------------------------------------
    # cast the country column to categorical:
    df['Country'] = df['Country'].astype('category')

    # plot the count plot for the country column:
    plt.figure(figsize=(10, 10))
    sns.countplot(x=df['Country'])
    plt.xticks(rotation=90)
    # give more space to the x_labels:
    plt.subplots_adjust(bottom=0.3)
    plt.title('Count plot for the country column')
    plt.savefig(Path('EDA', 'count_plot_country.png'))
    plt.close()

    # the dataset is dominated by UK customers, check the distribution without the UK:
    plt.figure(figsize=(10, 10))
    sns.countplot(x=df[df['Country'] != 'United Kingdom']['Country'])
    plt.xticks(rotation=90)
    # give more space to the x_labels:
    plt.subplots_adjust(bottom=0.3)
    plt.title('Count plot for the country column without UK')
    plt.savefig(Path('EDA', 'count_plot_country_without_uk.png'))
    plt.close()
    # We mostly have european customers, with australia being the largest non-european group.

    # Invoice number and date:
    # --------------------------------------------------------------
    # cast the invoice date column to datetime:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # plot the count plot for the invoice month column:
    plt.figure(figsize=(10, 10))
    sns.countplot(x=df['InvoiceDate'].dt.month)
    plt.title('Count plot for the invoice month column')
    plt.savefig(Path('EDA', 'count_plot_invoice_month.png'))
    plt.close()
    # we have more sales in november and december for this dataset, to consider when we decide the train/test split.

    # plot the count plot for the invoice day column:
    plt.figure(figsize=(10, 10))
    sns.countplot(x=df['InvoiceDate'].dt.day)
    plt.title('Count plot for the invoice day column')
    plt.savefig(Path('EDA', 'count_plot_invoice_day.png'))
    plt.close()
    # there is no clear pattern in the day of the month.

    # plot the count plot for the invoice day of week column:
    plt.figure(figsize=(10, 10))
    sns.countplot(x=df['InvoiceDate'].dt.dayofweek)
    plt.title('Count plot for the invoice day of week column')
    plt.savefig(Path('EDA', 'count_plot_invoice_day_of_week.png'))
    plt.close()
    # surprisingly for an online store, we have no sales on saturday, not sure how to interpret this.

    # plot the count plot for the invoice hour column:
    plt.figure(figsize=(10, 10))
    sns.countplot(x=df['InvoiceDate'].dt.hour)
    plt.title('Count plot for the invoice hour column')
    plt.savefig(Path('EDA', 'count_plot_invoice_hour.png'))
    plt.close()
    # we have a standard distribution of sales throughout the day, with a peak at 12:00.

    # import the cleaned dataset:
    # df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_cleaned.csv'))

    # create the aggregated costumer dataset:
    df_agg = df.groupby('InvoiceDate').agg(
        {'Invoice': 'count', 'Quantity': 'sum', 'Price': 'sum'})
    df_agg.rename(columns={'Invoice': 'NumberOfPurchases', 'Quantity': 'TotalQuantity', 'Price': 'TotalSpent'},
                  inplace=True)
    df_agg.index = pd.to_datetime(df_agg.index)

    # show the plots for number of purchases and total money spent
    x = df_agg.index

    plt.figure(figsize=(10, 10))
    plt.scatter(x=x, y=df_agg['NumberOfPurchases'], marker='.')
    mean_n_purch = np.mean(df_agg['NumberOfPurchases']).astype(int)
    npurch = df_agg['NumberOfPurchases']
    y_average1 = npurch.rolling(window=mean_n_purch).mean()
    plt.plot(x, y_average1, label='Rolling mean', linestyle='-', c='orange')  # mean line
    plt.xlabel('datetime')
    plt.ylabel('number of purchases')
    plt.ylim(0, 800)
    plt.legend()
    plt.grid(True)
    plt.savefig(Path('EDA', 'nr_purchases_trend.png'))
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.scatter(x, df_agg['TotalSpent'], marker='.')
    mean_tot_spend = np.mean(df_agg['TotalSpent']).astype(int)
    tot_spend = df_agg['TotalSpent']
    y_average2 = tot_spend.rolling(window=mean_tot_spend).mean()
    plt.plot(x, y_average2, label='Rolling mean', linestyle='-', c='orange')  # mean line
    plt.ylim(0, 10000)
    plt.xlabel('datetime')
    plt.ylabel('money spent')
    plt.legend()
    plt.grid(True)
    plt.savefig(Path('EDA', 'total_spent_trend.png'))
    plt.close()



