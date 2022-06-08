from pathlib import Path
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

#columns = (0, 1)
#loaded_data = load_and_shuffle_data(r'C:\Users\KSpicer\Documents\GitHub\victorian_authorship_classification_playground', 'training_data.csv', columns, 123)
#print(type(loaded_data))
def load_victorian_dataset(data_path, filename, cols, validation_split=0.2, seed=123):
    columns = cols
    data = load_and_shuffle_data(data_path, filename, columns, seed, header=0)
    texts = list(data['text'])
    labels = np.array(data['author'])
    #return texts, labels
    return (split_training_and_validation_sets(texts, labels, validation_split='validation_split'))

#print(len(test_texts))
#print(len(test_labels))
#print(test_texts[0])
#print("\n")
#print(test_labels[0])

def split_training_and_validation_sets(texts, labels, validation_split):
    num_training_samples = int((1 - validation_split) * len(texts))
    return ((texts[:num_training_samples], labels[:num_training_samples]),
            (texts[num_training_samples:], labels[num_training_samples:]))
    
#split_texts, split_labels = split_training_and_validation_sets(test_texts, test_labels, .2)
#x, y = split_texts
#print(len(y))
#
test_texts, test_labels = load_victorian_dataset(r'C:\Users\KSpicer\Documents\GitHub\victorian_authorship_classification_playground', 'training_data.csv', (0, 1), .2, 123)

#print('\n')
#print(type(split_texts))
#print(type(split_labels))
#print('\n')
#print(len(split_texts))
#print(len(split_labels))


#path =  Path('training_data.csv')
#data = load_victorian_dataset(path.parent, 'training_data.csv', )
#
#print("Successfully loaded data and dataset and shuffled it accordingly ...")
##print(data.shape)
##print(type(data))
##print(data.head(15))
#
#training_data, training_labels = split_training_and_validation_sets(data['text'], data['author'], 0.2)
##print(type(training_data))
##print(type(training_labels))
##print(training_data[0])
##print(training_data[1])
#print(training_labels[0])
##print(training_labels[1])
#
#NGRAM_RANGE = 2
#TOP_K = 20000
#TOKEN_MODE = 'word'
#MIN_DOCUMENT_FREQUENCY = 2
#MAX_SEQUENCE_LENGTH = 1000
