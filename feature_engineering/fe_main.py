import nltk

from feature_engineering.fe_aggregate_datasets import main as fe_aggreagtor
from feature_engineering.graph import customer_graph_creation, customer_country_graph_creation, product_graph_creation, \
    deepwalk, centrality_measures, graph_aggregate_dataset
from feature_engineering.nlp import nlp_feature_engineering, nlp_aggregate_datasets
from feature_engineering.rfm_base_model.base_dataset import main as base
from feature_engineering.time_series.ats_extractor import main as ats_extr
from feature_engineering.time_series.missing_values_analysis import main as missing_values


def check_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')


def embeddings():
    deepwalk.main('customer')
    deepwalk.main('customer_country')
    deepwalk.main('product')


def nlp():
    nlp_feature_engineering.main()
    nlp_aggregate_datasets.main()


def graph():
    print('> customer graph')
    customer_graph_creation.main()
    print('> customer_country graph')
    customer_country_graph_creation.main()
    print('> product graph')
    product_graph_creation.main()
    for _type in ['product', 'customer', 'customer_country']:
        print(f'> calculating centrality for {_type}')
        centrality_measures.main(_type, only_pagerank=False)
    print('> aggregating graph datasets')
    graph_aggregate_dataset.main()


def main():
    print('> checking nltk downloadable')
    check_nltk()
    print('\n> creating base dataset')
    base()
    print('\n> creating graph datasets')
    graph()
    print('\n> calculating embeddings')
    embeddings()
    print('\n> creating nlp datasets')
    nlp()
    print('\n> creating time series datasets')
    print(' /!\\ NOTE: After the first progress bar, the function will print lots of stuff. '
          'This is the normal behaviour. Please do not stop the process.')
    ats_extr()
    missing_values()
    print('\n> aggregating feature engineering datasets')
    fe_aggreagtor()


if __name__ == '__main__':
    main()
