# library to aggregate the extracted graph features:

import pandas as pd
from pathlib import Path

# Driver:
if __name__ == '__main__':
    # load the centrality measures:
    df_customer_centrality = pd.read_csv(Path('..', '..', 'data', 'customer_graph_centrality.csv'), index_col=0)
    df_product_centrality = pd.read_csv(Path('..', '..', 'data', 'product_graph_centrality.csv'), index_col=0)
    df_customer_country_centrality = pd.read_csv(Path('..', '..', 'data', 'customer_country_graph_centrality.csv'),
                                                 index_col=0)

    # load the deepwalk measures:
    df_customer_deepwalk = pd.read_csv(Path('..', '..', 'data', 'customer_graph_deepwalk.csv'), index_col=0)
    df_product_deepwalk = pd.read_csv(Path('..', '..', 'data', 'product_graph_deepwalk.csv'), index_col=0)
    df_customer_country_deepwalk = pd.read_csv(Path('..', '..', 'data', 'customer_country_graph_deepwalk.csv'),
                                               index_col=0)

    # merge the centrality and deepwalk measures:
    df_customer = pd.concat([df_customer_centrality, df_customer_deepwalk], axis=1)
    df_product = pd.concat([df_product_centrality, df_product_deepwalk], axis=1)
    df_customer_country = pd.concat([df_customer_country_centrality, df_customer_country_deepwalk], axis=1)

    # merge the customer and customer_country features:
    df_customer = pd.concat([df_customer, df_customer_country], axis=1)

    # save the extracted features to a csv file:
    df_customer.to_csv(Path('..', '..', 'data', 'customer_graph_features.csv'))
    df_product.to_csv(Path('..', '..', 'data', 'product_graph_features.csv'))

    # print the number of extracted features:
    print(f"Number of customer graph features: {df_customer.shape[1]}")
    print(f"Number of product graph features: {df_product.shape[1]}")

    # todo how do we merge the customer and product features?
