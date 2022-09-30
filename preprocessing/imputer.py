# 1st method is to drop the null rows so this .py is not useful for the moment

# Libraries:
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


# sponsored by stackoverflow:
class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


if __name__ == '__main__':
    df = pd.read_csv('../data/online_sales_dataset_description_imputed.csv', dtype={'StockCode': str})
    df = df.iloc[:10000, :]  # for testing the KNNImputer I sliced a tiny part, the complete df might be too large

    # encode the df and dropping columns that will not be used (also StockCode since gives problems with mixed dtypes)
    df_enc = MultiColumnLabelEncoder(columns=['Invoice', 'Description', 'Country'])\
        .fit_transform(df.drop(['InvoiceDate', 'StockCode', 'Price'], axis=1))
    print(df_enc.info())
    print(df_enc.head())
    print(df_enc.isnull().sum())

    imp = KNNImputer(n_neighbors=3)
    df_ft = imp.fit_transform(df_enc)
    df_knn = pd.DataFrame(df_ft, columns=imp.get_feature_names_out())

    print('\nFitted transformed dataframe:')
    print(df_knn.info())
    print(df_knn.head())
    print(df_knn.isnull().sum())

    df_knn.to_csv(Path('../data/online_sales_dataset_description_KNN.csv'), index=False)

