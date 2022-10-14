# Libraries:
import networkx as nx
import pandas as pd
from pathlib import Path


from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':
    # import the timeseries dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))

    # group by customer:
    df_c = df.groupby('CustomerId').agg({'StockCode': lambda x: list(x),
                                         'Country': lambda x: x.value_counts().index[0]})
    # cast customer id to int:
    df_c['CustomerId'] = df_c.index.astype(int)

    df_s = df.groupby('StockCode').agg({'Price': 'mean'})

    # cast stock code to int:
    df_s['StockCode'] = df_s.index.astype(int)

    # create a graph:
    G = nx.DiGraph()

    # add CustomerIds as nodes with blue color:
    G.add_nodes_from(df_c['CustomerId'], color='blue', size=5, country=df['Country'])
    G.add_nodes_from(df_s['StockCode'], color='red', size=5, price=df['Price'])

    # add edges, the weight is the price times the quantity of the product, if the edge already exists
    # add the weight to the existing weight:
    for row in df.itertuples():
        G.add_edge(row.CustomerId, row.StockCode, weight=row.Price * row.Quantity)

    # print the number of nodes:
    print(f'Number of nodes: {G.number_of_nodes()}')

    # print the number of edges:
    print(f'Number of edges: {G.number_of_edges()}')

    # save the graph with pylab:
    nx.draw(G, node_size=5, font_size=5)
    plt.savefig(Path('..', '..', 'plots', 'customer_product_graph.png'))
