# # Libraries:
# data manipulation:
import pandas as pd
from pathlib import Path
import pickle
import tabulate
# modelling:
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
from tuning.xgboost_tuner import tuner
from sklearn.model_selection import train_test_split

# Functions:
if __name__ == '__main__':
    # read the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # select the features: number of purchases, total price spent, total quantity ordered and country:
    X = df_agg[['NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country', 'Recency']]
    y = df_agg['CustomerChurned']

    # train test split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # tune the model with bayesian optimization:
    best_parameters = tuner(X_train, y_train, X_test, y_test)

    # print the best parameters:
    print('Best parameters:')
    print(best_parameters)

    # train the model with the best parameters:
    model = XGBClassifier(**best_parameters, objective="binary:logistic", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_predicted = model.predict(X_test)

    # print the classification report and the f1 score:
    print(classification_report(y_test, y_predicted))
    print(f"Tuned base model has an f-score of: {f1_score(y_test, y_predicted):.3f}")

    # print the feature importance:
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    # sort by importance:
    importance.sort_values(by='importance', ascending=False, inplace=True)
    # print the feature importance with tabulate:
    print(tabulate.tabulate(importance, headers='keys', tablefmt='psql'))

    # save the model as a pickle file:
    with open(Path('..', 'models', 'xgboost_base_model_tuned.pkl'), 'wb') as f:
        pickle.dump(model, f)
