import numpy as np
import os
import pandas as pd

def load_and_shuffle_data(data_path, file_name, cols, seed, separator=',', header=0):
    np.random.seed(seed)
    data_path = os.path.join(data_path, file_name)
    df = pd.read_csv(data_path, encoding='latin-1', usecols=cols, sep=separator, header=header)
    return df.reindex(np.random.permutation(df.index))
    
def load_victorian_dataset(data_path, validation_split=0.2, seed=123):
    columns = (0, 1)
    train_data = load_and_shuffle_data(data_path, columns, seed, header=0)
    test_data = pd.read_csv(testing_data_file, encoding='latin-1')
    
    texts = list(data['text'])
    labels = np.array(data['author'])
    return (split_training_and_validation_sets(texts, labels, validation_split))
    
def split_training_and_validation_sets(texts, labels, validation_split):
    """Splits the texts and labels into training and validation sets.
    # Arguments
        texts: list, text data.
        labels: list, label data.
        validation_split: float, percentage of data to use for validation.
    # Returns
        A tuple of training and validation data.
    """
    num_training_samples = int((1 - validation_split) * len(texts))
    return ((texts[:num_training_samples], labels[:num_training_samples]),
            (texts[num_training_samples:], labels[num_training_samples:]))
    
data = load_and_shuffle_data('/Users/spicy.kev/Desktop/victorian_authorship', 'victorian_training_data.csv', (0, 1), 123)

print(data.shape)
print(type(data))