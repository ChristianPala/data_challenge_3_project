# Auxiliary library to perform the exploratory data analysis on the online sales dataset.
from pathlib import Path

import numpy as np
import pandas as pd

# Libraries:
# Data manipulation:
from data_loading import load_and_save_data

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

    # Replace the comma with a dot in the price column:
    df['Price'] = df['Price'].astype(str).str.replace(',', '.')
    # Cast the price column to float:
    df['Price'] = df['Price'].astype(float)

    # check the min and max values for the price column:
    print(f"Min price: {df['Price'].min()}")
    print(f"Max price: {df['Price'].max()}")

    """
    We also have negative values for price. We will solve it in the data cleaning phase.
    """

    # check the missing values:
    print(df.isnull().sum())

    # Numerical variables:
    # --------------------------------------------------------------

    # plot the correlation matrix, excluding the CustomerId column:
    plt.figure(figsize=(5, 5))
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
    # check the number of unique values for the country column as a percentage of the total number of rows:
    print(f"Percentage of unique values for the country column: {df['Country'].nunique() / df.shape[0] * 100}")
    print(df['Country'].value_counts())

    # check the stock codes:
    print(f"Number of unique values for the stock code column: {df['StockCode'].nunique()}")
    print(df['StockCode'].value_counts())

    # plot the count plot for the country column:
    plt.figure(figsize=(10, 10))
    sns.countplot(x=df['Country'])
    plt.xticks(rotation=90)
    # give more space to the x_labels:
    plt.subplots_adjust(bottom=0.3)
    plt.title('Count plot for the country column')
    plt.savefig(Path('EDA', 'count_plot_country.png'))
    plt.close()

    # plot the sales of the top 10 countries:
    plt.figure(figsize=(7, 7))
    sns.countplot(x=df['Country'], order=df['Country'].value_counts().iloc[1:11].index, color='blue')
    plt.xticks(rotation=90)
    # give more space to the x_labels:
    plt.subplots_adjust(bottom=0.3)
    # y label:
    plt.ylabel('Number of sales')
    plt.title('Top 10 sales by country, excluding the UK')
    plt.savefig(Path('EDA', 'count_plot_country_top_10.png'))

    # Invoice number and date:
    # --------------------------------------------------------------
    # cast the invoice date column to datetime:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # plot the count plot for the invoice month column:
    plt.figure(figsize=(7, 7))
    sns.countplot(x=df['InvoiceDate'].dt.month, color='blue')
    plt.title('Yearly number of sales')
    plt.xticks(np.arange(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.savefig(Path('EDA', 'count_plot_invoice_month.png'))
    plt.close()
    # we have more sales in november and december for this dataset, to consider when we decide the train/test split.

    # plot the count plot for the invoice day column:
    plt.figure(figsize=(7, 7))
    sns.countplot(x=df['InvoiceDate'].dt.day, color='blue')
    plt.title('Monthly number of sales')
    plt.savefig(Path('EDA', 'count_plot_invoice_day.png'))
    plt.close()
    # there is no clear pattern in the day of the month.

    # plot the count plot for the invoice day of week column, change the labels to the day of week:
    plt.figure(figsize=(7, 7))
    sns.countplot(x=df['InvoiceDate'].dt.dayofweek, color='blue')
    plt.title('Weekly number of sales')
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],)
    plt.savefig(Path('EDA', 'count_plot_invoice_day_of_week.png'))
    plt.close()
    # surprisingly for an online store, we have no sales on saturday, not sure how to interpret this.

    # plot the count plot for the invoice hour column:
    plt.figure(figsize=(7, 7))
    sns.countplot(x=df['InvoiceDate'].dt.hour, color='blue')
    plt.title('Hourly number of sales')
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
