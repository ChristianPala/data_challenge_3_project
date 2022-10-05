# Libraries:
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from pathlib import Path
import pandas as pd

if __name__ == '__main__':

    # import the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # create a standard tokenizer:
    # the relevant words are arbitrarily set to the top 1000 most frequent words:
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    # out of vocabulary should be irrelevant here, but it is good practice to set it anyway.
    tokenizer.fit_on_texts(df_agg['Description'])
    word_index = tokenizer.word_index
    removable = stopwords.words('english')

    # I first added and not k.isnumeric(), but maybe we want to keep the numbers?
    word_index = {k: v for k, v in word_index.items() if k not in removable}

    word_index_list = list(word_index.keys())

    # filter the aggregated descriptions leaving words in the word index list:
    df_agg['Description'] = df_agg['Description'].apply(lambda x: set(''.join([word for word in x.split()
                                                        if word in word_index_list])))
    # save the dataset:a
    df_agg.to_csv(Path('..', 'data', 'online_sales_dataset_agg_nlp.csv'), index=False)

    print(word_index)
