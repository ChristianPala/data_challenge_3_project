# library to aggregate the extracted graph features:
import pandas as pd
from pathlib import Path
import numpy as np

# Driver:
if __name__ == '__main__':
    # load the centrality measures:
    df_customer_centrality = pd.read_csv(Path('..', '..', 'data', 'customer_graph_centrality.csv'), index_col=0)
    df_product_centrality = pd.read_csv(Path('..', '..', 'data', 'product_graph_centrality.csv'), index_col=0)
    df_customer_country_centrality = pd.read_csv(Path('..', '..', 'data', 'customer_country_graph_centrality.csv'),
                                                 index_col=0)

    # load the deepwalk measures:
    df_customer_deepwalk = pd.read_csv(Path('..', '..', 'data', 'customer_deepwalk_embeddings.csv'), index_col=0)
    df_customer_country_deepwalk = pd.read_csv(Path('..', '..', 'data', 'customer_country_deepwalk_embeddings.csv'),
                                               index_col=0)

    # merge the centrality and deepwalk measures:
    df_customer = pd.concat([df_customer_centrality, df_customer_deepwalk], axis=1)
    df_customer_country = pd.concat([df_customer_country_centrality, df_customer_country_deepwalk], axis=1)

    # add _country to all the columns of the customer_country dataframe:
    df_customer_country.columns = [f'{col}_country' for col in df_customer_country.columns]

    # merge the customer and customer_country features:
    df_customer = pd.concat([df_customer, df_customer_country], axis=1)

    # check for missing values:
    print(f"Number of missing values in customer graph features: {df_customer.isnull().sum().sum()}")
    print(f"Number of missing values in product graph features: {df_product_centrality.isnull().sum().sum()}")

    # import the dataset for feature engineering:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))

    # create a list of customers with their product purchases:
    df = df.groupby('CustomerId').agg({'StockCode': lambda x: list(x)})

    # based on the product purchases, create an average value of the pagerank for each customer:
    df['ProductPagerank'] = df['StockCode'].apply(lambda x: np.mean([df_product_centrality.loc[product, 'pagerank']
                                                                     for product in x]))

    # merge the customer graph features with the dataset:
    df = df.merge(df_customer, how='left', left_index=True, right_index=True)

    # drop the StockCode column:
    df.drop('StockCode', axis=1, inplace=True)

    # save the dataset:
    df.to_csv(Path('..', '..', 'data', 'online_sales_dataset_graph_for_fs.csv'))
