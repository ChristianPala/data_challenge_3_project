# Libraries:
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':
    # read the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # select the features: number of purchases, total price spent, total quantity ordered and country:
    X = df_agg[['NumberOfPurchases', 'TotalSpent', 'TotalQuantity', 'Country']]
    y = df_agg['CustomerChurned']

    # train test split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the model:
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    # predict:
    y_pred = model.predict(X_test)

    # evaluate:
    print(classification_report(y_test, y_pred))
    print(f1_score(y_test, y_pred))

    # visualize initial features importance:
    importance = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    # sort by importance:
    importance.sort_values(by='importance', ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    plt.bar(importance['feature'], importance['importance'])
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()

