# Auxiliary function to split the data into training, validation and test sets consistently:

# Libraries:
import pandas as pd
from sklearn.model_selection import train_test_split


def train_validation_test_split(features: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Splits the data into training, validation and test sets.
    @param features: features.
    @param y: target.
    :return: tuple containing the three sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    # in case we want to add validation set:
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    return X_train, X_test, y_train, y_test
