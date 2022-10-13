# Libraries:
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# Functions:
def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculates the Jaccard index between two sets.
    @param set1: first set.
    @param set2: second set.
    :return: Jaccard index.
    """
    return len(set1.intersection(set2)) / len(set1.union(set2))


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


if __name__ == '__main__':

    # import the aggregated dataset:
    df_agg = pd.read_csv(Path('../..', 'data', 'online_sales_dataset_agg.csv'))

    # tokenize the description column:
    df_agg['Description'] = df_agg['Description'].apply(word_tokenize)

    # remove stopwords and punctuation:
    punctuation = ['.', ',', '!', '?', '(', ')', '[', ']', '{', '}', ':', ';', '"', "'"]
    stop_words = stopwords.words('english') + punctuation
    df_agg['Description'] = df_agg['Description'].apply(lambda x: [word for word in x if word not in stop_words])

    # remove duplicate words from the tokenized description:
    df_agg['Description'] = df_agg['Description'].apply(lambda x: set(x))

    # split the data set, note should stay consistent with all other splits!
    X_train, X_test, y_train, y_test = train_validation_test_split(df_agg.drop('CustomerChurned', axis=1),
                                                                   df_agg['CustomerChurned'])

    # cluster:
    df_clusters = pd.DataFrame(columns=['CustomerId', 'ClusterId', 'Description'])
    similarity_threshold = 0.8

    for index, row in X_train.iterrows():
        similarity = []
        # save all the similarity scores in a list
        for index2, row2 in X_train.iterrows():
            similarity.append(jaccard_similarity(row['Description'], row2['Description']))
        # find the indexes of the rows that have similarity score bigger than the similarity threshold
        indexes = [i for i, x in enumerate(similarity) if x >= similarity_threshold]
        # if there is more than one row with similarity score bigger than the threshold:
        if len(indexes) > 1:
            # add the row to the new data frame and the shared description to the cluster:
            df_clusters = pd.concat([df_clusters, pd.DataFrame(
                [[row['CustomerId'], ','.join([str(X_train.iloc[i]['CustomerId']) for i in indexes])]],
                columns=['CustomerId', 'ClusterId'])], ignore_index=True)

    # for each cluster, compute the shared description:
    df_clusters['Description'] = df_clusters['ClusterId']\
        .apply(lambda x: set.union(*[X_train.loc[X_train['CustomerId'] == int(float(i))]['Description'].iloc[0]
                                     for i in x.split(',')]))

    # save the dataset:
    X_train.to_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_X_train.csv'), index=False)
    df_clusters.to_csv(Path('..', '..', 'data', f'online_sales_dataset_clusters_{similarity_threshold}.csv'),
                       index=False)
