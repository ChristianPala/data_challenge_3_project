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

    # create a standard tokenizer and see what we get from the descriptions:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Description'])
    word_index = tokenizer.word_index
    removable = stopwords.words('english')
    # I first added and not k.isnumeric(), but maybe we want to keep the numbers?
    word_index = {k: v for k, v in word_index.items() if k not in removable}
    print(word_index)