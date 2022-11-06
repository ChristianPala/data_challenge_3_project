from feature_engineering_performances.base_model import main as base_model
from feature_engineering_performances.base_model_tuning import main as base_model_tuned
from feature_engineering_performances.base_model_cross_validation import main as base_model_cv
from feature_engineering_performances.ats_model import main as time_series_model
from feature_engineering_performances.nlp_model import main as nlp_model
from feature_engineering_performances.graph_model import main as graph_model


def main():
    base_model()
    base_model_cv()
    base_model_tuned()
    time_series_model()
    nlp_model()
    graph_model()


if __name__ == '__main__':
    main()
