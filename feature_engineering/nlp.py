# Libraries:
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from pathlib import Path
import pandas as pd
from sklearn.metrics import jaccard_score

if __name__ == '__main__':

    # import the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # create a standard tokens
    tokens = word_tokenize(df_agg['Description'].str.cat(sep=' '), language='english')

    # filter them:
    removable = stopwords.words('english') + ['.', ',', '!', '?', '(', ')', '[', ']', '{', '}',
                                              ':', ';', "'", '"', '``', "''"] + ['£', '€', '$']
    cleaned_tokens = [token for token in tokens if token not in removable]

    # add the word frequency:
    f_dist = FreqDist(cleaned_tokens)

    # compact into a dictionary the key is the word position by frequency and the value is the word:
    word_index = {i: word for i, word in enumerate(f_dist.keys())}

    word_index_list = list(word_index.values())

    # filter the aggregated descriptions leaving words in the word index list:
    df_agg['Description'] = df_agg['Description'].apply(lambda x: set(''.join([word for word in x.split()
                                                        if word in word_index_list])))
    print(word_index_list)

    # save the dataset:
    df_agg.to_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp.csv'), index=False)

    # use jaccard to calculate similarity between descriptions:

