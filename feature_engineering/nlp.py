# Libraries:
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    # import the cleaned dataset:
    df = pd.read_csv(Path('..', 'data', 'online_sales_dataset_cleaned.csv'))

    # import the aggregated dataset:
    df_agg = pd.read_csv(Path('..', 'data', 'online_sales_dataset_agg.csv'))

    # aggregate all descriptions belonging to the same customer:
    df_agg['Description'] = df.groupby('CustomerId')['Description'].transform(lambda x: ' '.join(x))

    # create a standard tokenizer:
    # the relevant words are arbitrarily set to the top 1000 most frequent words:
    tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
    # out of vocabulary should be irrelevant here but it is good practice to set it anyway.
    tokenizer.fit_on_texts(df_agg['Description'])
    word_index = tokenizer.word_index
    removable = stopwords.words('english')

    # I first added and not k.isnumeric(), but maybe we want to keep the numbers?
    word_index = {k: v for k, v in word_index.items() if k not in removable}

    print(word_index)