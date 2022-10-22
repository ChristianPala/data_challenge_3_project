# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling:
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split
from xgboost import XGBClassifier
from tuning.xgboost_tuner import tuner
from reporting.classifier_report import report_model_results


if __name__ == '__main__':
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_tsfel.csv'), header=0)