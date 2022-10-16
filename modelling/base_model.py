# Libraries:
import pandas as pd
from tabulate import tabulate
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from modelling.data_splitting.train_val_test_splitter import train_validation_test_split

matplotlib.use('TkAgg')

if __name__ == '__main__':
    # read the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # select the features: number of purchases, total price spent, total quantity ordered and country:
    X = df_agg[['Recency', 'NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]
    y = df_agg['CustomerChurned']

    # train test split:
    X_train, X_test, y_train, y_test = train_validation_test_split(X, y)

    # define the model:
    model = XGBClassifier(objective="binary:logistic", n_estimators=500, random_state=42)

    # fit the model:
    model.fit(X_train, y_train)

    # predict:
    y_pred = model.predict(X_test)

    # evaluate:
    print(classification_report(y_test, y_pred))
    print(f"f-score for the base model: {f1_score(y_test, y_pred): .3f}")

    # visualize initial features importance:
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    # sort by importance:
    importance.sort_values(by='importance', ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    plt.bar(importance['feature'], importance['importance'])
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    # save the figure:
    plt.savefig(Path('..', 'plots', 'feature_importance_RFM.png'))

    # print the feature importance in a table:
    print(tabulate(importance, headers='keys', tablefmt='psql'))