import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

def load_and_shuffle_data(data_path, filename, cols, seed, separator=',', header=0):
    np.random.seed(seed)
    data_path = os.path.join(data_path, filename)
    df = pd.read_csv(data_path, encoding='latin-1', usecols=cols, sep=separator, header=header)
    return df.reindex(np.random.permutation(df.index))

def load_victorian_dataset(data_path, filename, cols, validation_split=0.2, seed=123):
    columns = cols
    data = load_and_shuffle_data(data_path, filename, columns, seed, header=0)
    texts = list(data['text'])
    labels = np.array(data['author'])
    #return texts, labels
    return (split_training_and_validation_sets(texts, labels, validation_split=validation_split)) # for the longest time I had this as validation_split='validation_split' thus raising an error ...

def split_training_and_validation_sets(texts, labels, validation_split):
    num_training_samples = int((1 - validation_split) * len(texts))
    return ((texts[:num_training_samples], labels[:num_training_samples]),
            (texts[num_training_samples:], labels[num_training_samples:]))
