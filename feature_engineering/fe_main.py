from rfm_base_model.base_dataset import main as base
from time_series.ats_extractor import main as ats_extr
from time_series.missing_values_analysis import main as missing_values
from nlp import nlp_feature_engineering, nlp_aggregate_datasets
from graph import customer_graph_creation, customer_country_graph_creation, product_graph_creation, deepwalk, centrality_measures
from fe_aggregate_datasets import main as fe_aggreagtor
import nltk


def check_nltk():
    print('checking missing libraries and options')
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
    customer_graph_creation.main()
    customer_country_graph_creation.main()
    product_graph_creation.main()
    centrality_measures.main('product')
    centrality_measures.main('customer')
    centrality_measures.main('customer_country')


def main():
    check_nltk()
    base()
    graph()
    embeddings()
    nlp()
    ats_extr()
    missing_values()
    fe_aggreagtor()


if __name__ == '__main__':
    main()
