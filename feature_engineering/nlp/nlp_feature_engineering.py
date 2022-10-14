# Libraries:
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# global variables:
similarity_threshold = 0.7


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


def print_results(training_set: pd.DataFrame, testing_set: pd.DataFrame) -> None:
    """
    Prints the results of the analysis.
    @param training_set: training set.
    @param testing_set: testing set.
    :return: None.
    """

    # test the minimum and maximum values of the Cluster_i features for the train and test sets:
    print(training_set.filter(regex='Cd_').min().min())
    print(training_set.filter(regex='Cd_').max().max())
    print(testing_set.filter(regex='Cd_').min().min())
    print(testing_set.filter(regex='Cd_').max().max())

    # count how many customers are in each cluster for both train and test sets:
    print(training_set.filter(regex='Cd_').sum().sum())
    print(testing_set.filter(regex='Cd_').sum().sum())


def tokenize_description(df: pd.DataFrame) -> pd.DataFrame:
    # tokenize the description column:
    df['Description'] = df['Description'].apply(word_tokenize)

    # remove stopwords and punctuation:
    punctuation = ['.', ',', '!', '?', '(', ')', '[', ']', '{', '}', ':', ';', '"', "'"]
    stop_words = stopwords.words('english') + punctuation
    df['Description'] = df['Description'].apply(lambda x: [word for word in x if word not in stop_words])

    # remove duplicate words from the tokenized description:
    df['Description'] = df['Description'].apply(lambda x: set(x))

    return df


def apply_clustering_to_dataset(training_set: pd.DataFrame, testing_set: pd.DataFrame,
                                df_cluster: pd.DataFrame) -> None:
    """
    Applies the clustering to the training and testing datasets.
    For each cluster create a feature in the train dataset, that indicates if the customer is in the cluster:
    For each cluster create a feature in the test dataset, that indicates if the customer is in the cluster,
    the assignment is based on the jaccard similarity of the description of the customer and cluster:
    :param training_set: the training set features as a pandas DataFrame.
    :param testing_set: the testing set features as a pandas DataFrame.
    :param df_cluster:  the clusters as a pandas DataFrame.
    :return: None.
    """

    for index, row in df_cluster.iterrows():
        training_set['Cd_' + str(index)] = training_set['CustomerId'].apply(lambda x: 1 if x in row['ClusterId'] else 0)
        testing_set['Cd_' + str(index)] = testing_set['Description']\
            .apply(lambda x: 1 if jaccard_similarity(x, row['Description']) > similarity_threshold else 0)

    # Save the elaborated datasets:
    df_cluster.to_csv(Path('..', '..', 'data', f'online_sales_dataset_clusters_{similarity_threshold}.csv'),
                      index=False)
    training_set.to_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_train.csv'), index=False)
    testing_set.to_csv(Path('..', '..', 'data', 'online_sales_dataset_agg_nlp_test.csv'), index=False)


def create_clusters_dataframe(df: pd.DataFrame) -> None:
    """
    Creates clusters of similar descriptions in the training set and adds them as features to the
    training and testing datasets, return the modified training and testing datasets and the clusters.
    :param df: the aggregated dataset.
    :return: training set, testing set and clusters as pandas DataFrames.
    """
    # cluster:
    df_cluster = pd.DataFrame(columns=['CustomerId', 'ClusterId', 'Description'])

    # split the data set, note should stay consistent with all other splits!
    training_features, testing_features, y_train, y_test = train_validation_test_split(df.drop('CustomerChurned',
                                                                                               axis=1),
                                                                                       df['CustomerChurned'])
    for index, row in training_features.iterrows():
        similarity = []
        # save all the similarity scores in a list
        for index2, row2 in training_features.iterrows():
            similarity.append(jaccard_similarity(row['Description'], row2['Description']))
        # find the indexes of the rows that have similarity score bigger than the similarity threshold
        indexes = [i for i, x in enumerate(similarity) if x >= similarity_threshold]
        # if there is more than one row with similarity score bigger than the threshold:
        if len(indexes) > 1:
            # add the row to the new data frame and the shared description to the cluster:
            df_clusters = pd.concat([df_cluster, pd.DataFrame(
                [[row['CustomerId'], ','.join([str(training_features.iloc[i]['CustomerId']) for i in indexes])]],
                columns=['CustomerId', 'ClusterId'])], ignore_index=True)

    # for each cluster, compute the shared description:
    df_cluster['Description'] = df_cluster['ClusterId']\
        .apply(lambda x: set.union(*[training_features.loc[training_features['CustomerId'] == int(float(i))]
                                                                     ['Description'].iloc[0]
                                     for i in x.split(',')]))

    # cast the ClusterId as a set of floats:
    df_cluster['ClusterId'] = df_cluster['ClusterId'].apply(lambda x: set([float(i) for i in x[0:-1].split(',')]))
    df_cluster['ClusterSize'] = df_cluster['ClusterId'].apply(lambda x: len(x))

    # remove the redundant CustomerId column:
    df_cluster.drop('CustomerId', axis=1, inplace=True)

    # Keep only the first row of each cluster id:
    df_cluster.drop_duplicates(subset='ClusterId', keep='first', inplace=True)

    # For each cluster, check the purity with the CustomerChurned column:
    df_cluster['Churned'] = df_cluster['ClusterId'].apply(lambda x: df_agg[df_agg['CustomerId'].isin(x)]
    ['CustomerChurned'].sum())
    df_cluster['Churned'] = df_cluster['Churned'] / df_cluster['ClusterSize']
    # create a new column with the churned purity, 1 if all churned or did not churn, 0 otherwise:
    df_cluster['ClusterPurity'] = df_cluster['Churned'].apply(lambda x: 1 if x == 0 or x == 1 else 0)

    # apply the clustering to the training and testing datasets:
    apply_clustering_to_dataset(training_features, testing_features, df_cluster)

    # print the results:
    print_results(training_features, testing_features)


def main(dataframe: pd.DataFrame) -> None:
    """
    Main routine for the NLP features engineering library:
    :return: None
    """
    # tokenize the description column:
    df = tokenize_description(dataframe)
    # cluster and apply the clusters to the datasets:
    create_clusters_dataframe(df)


if __name__ == '__main__':
    # Load the  dataset:
    df_agg = pd.read_csv(Path('..', '..', 'data', 'online_sales_dataset_agg.csv'))
    # Process the dataset:
    main(df_agg)

