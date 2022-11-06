# Libraries
from preprocessing.churn_analysis import main as churn_analysis
from preprocessing.exploratory_data_analysis import main as eda
from preprocessing.preprocessing import main as preprocessing


def main(preprocessing_only=True):
    if not preprocessing_only:
        eda()
        preprocessing()
        churn_analysis()
    else:
        preprocessing()


if __name__ == '__main__':
    main(preprocessing_only=True)
