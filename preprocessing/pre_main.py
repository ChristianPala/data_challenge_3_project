# Libraries
from preprocessing import main as preprocessing
from churn_analysis import main as churn_analysis
from exploratory_data_analysis import main as eda


def main(preprocessing_only=True):
    if not preprocessing_only:
        eda()
        preprocessing()
        churn_analysis()
    else:
        preprocessing()


if __name__ == '__main__':
    main(preprocessing_only=True)

