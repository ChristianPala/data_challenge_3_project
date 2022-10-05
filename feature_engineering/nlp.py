# Libraries:
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from pathlib import Path
import pandas as pd


# Functions:
def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculates the Jaccard index between two sets.
    @param set1: first set.
    @param set2: second set.
    :return: Jaccard index.
    """
    return len(set1.intersection(set2)) / len(set1.union(set2))


if __name__ == '__main__':

    # import the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # tokenize the description column:
    df_agg['Description'] = df_agg['Description'].apply(word_tokenize)

    # remove stopwords and punctuation:
    punctuation = ['.', ',', '!', '?', '(', ')', '[', ']', '{', '}', ':', ';', '"', "'"]
    stop_words = stopwords.words('english') + punctuation
    df_agg['Description'] = df_agg['Description'].apply(lambda x: [word for word in x if word not in stop_words])

    # remove duplicate words from the tokenized description:
    df_agg['Description'] = df_agg['Description'].apply(lambda x: set(x))

    # cluster:
    df_clusters = pd.DataFrame(columns=['CustomerId', 'ClusterId'])
    similarity_threshold = 0.5

    for index, row in df_agg.iterrows():
        similarity = []
        # save all the similarity scores in a list
        for index2, row2 in df_agg.iterrows():
            similarity.append(jaccard_similarity(row['Description'], row2['Description']))
        # find the indexes of the rows that have similarity score bigger than the similarity threshold
        indexes = [i for i, x in enumerate(similarity) if x >= similarity_threshold]
        # if there is more than one row with similarity score bigger than the threshold:
        if len(indexes) > 1:
            # add the row to the new data frame
            df_clusters = pd.concat([df_clusters, pd.DataFrame(
                [[row['CustomerId'], ','.join([str(df_agg.iloc[i]['CustomerId']) for i in indexes])]],
                columns=['CustomerId', 'ClusterId'])], ignore_index=True)

    # save the dataset:
    df_agg.to_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp.csv'), index=False)
    df_clusters.to_csv(Path('..', 'data', f'online_sales_dataset_clusters_{similarity_threshold}.csv'), index=False)

