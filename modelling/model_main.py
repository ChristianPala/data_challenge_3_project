from modelling.feature_engineering_performances.ats_model import main as time_series_model
from modelling.feature_engineering_performances.base_model import main as base_model
from modelling.feature_engineering_performances.base_model_cross_validation import main as base_model_cv
from modelling.feature_engineering_performances.base_model_tuning import main as base_model_tuned
from modelling.feature_engineering_performances.graph_model import main as graph_model
from modelling.feature_engineering_performances.nlp_model import main as nlp_model


def main():
    print('\n> Running base model')
    base_model()
    print('\n> Running base model with cross validation')
    base_model_cv()
    print('\n> Running base model tuned')
    base_model_tuned()
    print('\n> Running time series model')
    time_series_model()
    print('\n> Running NLP model')
    nlp_model()
    print('\n> Running graph model')
    graph_model()


if __name__ == '__main__':
    main()
