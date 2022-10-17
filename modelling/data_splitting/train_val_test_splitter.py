# Auxiliary function to split the data into training, validation and test sets consistently:

# Libraries:
import pandas as pd
from sklearn.model_selection import train_test_split


def train_validation_test_split(features: pd.DataFrame, y: pd.Series, validation: bool = False) -> tuple:
    """
    Splits the data into training, validation and test sets.
    @param features: features.
    @param y: target.
    @param validation: bool: whether to split the data into training, validation and test sets.
    :return: tuple containing the three sets.
    """

    # in case we want to add validation set:
    if validation:
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
