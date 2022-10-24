# Select features using mutual information:

# Data manipulation:
import pandas as pd
from pathlib import Path
import tabulate

# Modelling:
from sklearn.feature_selection import mutual_info_classif
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split


# Driver:
if __name__ == '__main__':
    # import the RFM dataset:
    # todo: change this to the dataset with all the engineered features
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))

    y = df['CustomerChurned']

    # keep only the RFM features, change later to all the engineered features:
    X = df[['Recency', 'NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]

    # split the dataset into train and test:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)

    # calculate the mutual information:
    # mi ranges from 0 to 1, 0 means no mutual information, 1 means perfect mutual information
    mi = mutual_info_classif(X_train, y_train)

    # create a dataframe with the results:
    mi_df = pd.DataFrame({'feature': X_train.columns, 'MutualInformation': mi})

    # sort the dataframe:
    mi_df.sort_values('MutualInformation', ascending=False, inplace=True)

    # save the results to a csv file:
    mi_df.to_csv(Path('..', '..', 'data', 'feature_selection_mutual_information.csv'), index=False)

    # select features with mutual information a threshold of  0.0001:
    mi_threshold = 10 ** -4
    mi_df = mi_df[mi_df['MutualInformation'] >= mi_threshold]

    # print the results using tabulate as a percentage:
    print(tabulate.tabulate(mi_df, headers='keys', tablefmt='psql', showindex=False, floatfmt=".3%"))

