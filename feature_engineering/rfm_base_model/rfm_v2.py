# Aggregate the  by customer for the base model, use the RFM strategy to select features.
# Libraries:
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib

matplotlib.use('tkagg')


def customer_recency(data):
    # Recency
    max_date = data['InvoiceDate'].max()
    recency = data[data['InvoiceDate'] < cut_off].copy()  # Group customers by latest transaction
    recency = recency.groupby('CustomerId')['InvoiceDate'].max()
    return (max_date - recency).dt.days.reset_index().rename(columns={'InvoiceDate': 'recency'})


def customer_frequency(data):
    # Frequency
    frequency = data[data['InvoiceDate'] < cut_off].copy()  # Set date column as index
    frequency.set_index('InvoiceDate', inplace=True)
    frequency.index = pd.DatetimeIndex(frequency.index)  # Group transactions by customer key and by distinct period
    # and count transactions in each period
    frequency = frequency.groupby(['CustomerId', pd.Grouper(freq="M",
                                                            level='InvoiceDate')]).count()  # (Optional) Only count the number of distinct periods a transaction # occurred. Else, we will be calculating total transactions in each # period instead.
    frequency['Quantity'] = 1  # Store all distinct transactions
    # Sum transactions
    return frequency.groupby('CustomerId').sum().reset_index().rename(columns={'Quantity': 'frequency'})


def customer_value(data):
    # Monetary value
    value = data[data['InvoiceDate'] < cut_off].copy()  # Set date column as index
    value.set_index('InvoiceDate', inplace=True)
    value.index = pd.DatetimeIndex(value.index)
    # Get mean or total sales amount for each customer
    return value.groupby('CustomerId')['Price'].mean().reset_index().rename(columns={'Price': 'value'})


def customer_age(data):
    # AGE
    age = data[data['InvoiceDate'] < cut_off].copy()  # Get date of first transaction
    first_purchase = age.groupby('CustomerId')[
        'InvoiceDate'].min().reset_index()  # Get number of days between cut off and first transaction
    first_purchase['age'] = (cut_off - first_purchase['InvoiceDate']).dt.days
    return first_purchase


def customer_rfm(data, customer_id_column):
    recency = customer_recency(data)
    frequency = customer_frequency(data)
    value = customer_value(data)
    age = customer_age(data)
    # Merge all columns
    return recency.merge(frequency, on=customer_id_column).merge(value, on=customer_id_column).merge(age, on=customer_id_column)


def generate_churn_labels(future):
    future['CustomerChurned'] = 0
    return future[['CustomerId', 'CustomerChurned']]


def recursive_rfm(data, freq='M', start_length=30, label_period_days=30):  # Resultant list of datasets
    dset_list = []  # Get start and end dates of dataset
    date_col = 'InvoiceDate'
    id_col = 'CustomerId'
    value_col = 'Price'
    start_date = data[date_col].min() + pd.Timedelta(start_length, unit="D")
    end_date = data[date_col].max() - pd.Timedelta(label_period_days, unit="D")  # Get dates at desired interval
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    # data[date_col] = pd.to_datetime(data[date_col])
    for c in dates:
        # split by observed / future
        observed = data[data[date_col] < c]
        future = data[
            (data[date_col] > c) &
            (data[date_col] < c + pd.Timedelta(
                label_period_days, unit='D'))
            ]
        # Get relevant columns
        rfm_columns = [date_col, id_col, value_col]
        _observed = observed[rfm_columns]  # Compute features from observed
        rfm_features = customer_rfm(_observed, id_col)  # Set label for everyone who bought in 'future' as 0'
        labels = generate_churn_labels(future)  # Outer join features with labels to ensure customers
        # not in observed are still recorded with a label of 1
        dset = rfm_features.merge(labels, on=id_col, how='outer').fillna(1)
        dset_list.append(dset)  # Concatenate all datasets
    full_dataset = pd.concat(dset_list, axis=0)

    res = full_dataset[full_dataset.recency != 0].dropna(axis=1, how='any')

    return res


# Driver:
if __name__ == '__main__':
    # import the cleaned dataset:
    df = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_for_fe.csv'))
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    cut_off = df['InvoiceDate'].max() - pd.Timedelta('300')

    rec_df = recursive_rfm(df)
    rec_df = rec_df.sample(frac=1)  # Shuffle
    # print(rec_df.info())

    # Set X and y
    X = rec_df[['recency', 'frequency', 'value', 'age']]
    y = rec_df[['CustomerChurned']].values.reshape(-1)
    # Set test ratio and perform train / test split
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(rec_df['CustomerChurned'].value_counts())

    oversample = RandomOverSampler()
    X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

    # Initialize and fit model on train dataset
    rf = RandomForestClassifier().fit(X_train, y_train)
    # Fit on over-sampled data as well
    rf_over = RandomForestClassifier().fit(X_train_over, y_train_over)

    # Create Dataframe and populate with predictions and actuals
    # Train set
    predictions = pd.DataFrame()
    predictions['true'] = y_train
    predictions['preds'] = rf.predict(X_train)

    # Test set
    predictions_test = pd.DataFrame()
    predictions_test['true'] = y_test
    predictions_test['preds'] = rf.predict(X_test)
    predictions_test['preds_over'] = rf_over.predict(X_test)

    # Compute error
    train_acc = accuracy_score(predictions.true, predictions.preds)
    test_acc = accuracy_score(predictions_test.true, predictions_test.preds)
    test_acc_over = accuracy_score(predictions_test.true, predictions_test.preds_over)

    predictions_test['proba'] = rf.predict_proba(X_test)[:, 1]
    predictions_test['proba_over'] = rf_over.predict_proba(X_test)[:, 1]

    print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Test Acc Oversampled: {test_acc_over:.4f}")
    print(classification_report(predictions_test.true, predictions_test.preds))
    print(classification_report(predictions_test.true, predictions_test.preds_over))

    # plt.figure(figsize=(15, 7))
    plt.hist(predictions_test['proba'][y_test == 0], bins=50, label='Negatives')
    plt.hist(predictions_test['proba'][y_test == 1], bins=50, label='Positives', alpha=0.7, color='r')
    plt.xlabel('Probability of being Positive Class')
    plt.ylabel('Number of records in each bucket')
    plt.legend()
    # plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.yscale('log')
    plt.savefig(Path('..', '..', 'plots', 'probabilities'))
    plt.close()

    # plt.figure(figsize=(15, 7))
    plt.hist(predictions_test['proba_over'][y_test == 0], bins=50, label='Negatives')
    plt.hist(predictions_test['proba_over'][y_test == 1], bins=50, label='Positives', alpha=0.7, color='r')
    plt.xlabel('Probability of being Positive Class')
    plt.ylabel('Number of records in each bucket')
    plt.legend()
    # plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.yscale('log')
    plt.savefig(Path('..', '..', 'plots', 'probabilities oversampled'))

    rec_df.to_csv(Path('..', '..', 'data', 'online_sales_dataset_rec_rfm.csv'), index=False)
