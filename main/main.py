# Libraries
from preprocessing.pre_main import main as preprocessing
from feature_engineering.fe_main import main as feature_engineering
from feature_selection.fs_main import main as feature_selection
from dimensionality_reduction.dr_main import main as dimensionality_reduction
import nltk


def check_nltk():
    print('checking missing libraries and options')
    nltk.download('punkt')
    nltk.download('stopwords')


def main():
    check_nltk()
    # running preprocessing (includes data loading and other)
    preprocessing(preprocessing_only=True)
    feature_engineering()
    feature_selection(run_wrappers=False, run_correlation=False)
    dimensionality_reduction(run_full=False, run_dsa=False)


if __name__ == '__main__':
    main()
