from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import load_and_shuffle_data

NGRAM_RANGE = (1, 2)
TOP_K = 20000
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 2

def ngram_vectorizer(train_texts, train_labels, val_texts):
        kwargs = {
            'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': TOKEN_MODE,  # Split text into word tokens.
            'min_df': MIN_DOCUMENT_FREQUENCY,
        }
    
        vectorizer = TfidfVectorizer(**kwargs)
    
        X_train = vectorizer.fit_transform(train_texts)

        X_val = vectorizer.transform(val_texts)

        selector = SelectKBest(f_classif, k=min(TOP_K, X_train.shape[1]))
        selector.fit(X_train, train_labels)
        X_train = selector.transform(X_train).astype('float32')
        X_val = selector.transform(X_val).astype('float32')

        return X_train, X_val

def sequence_vectorizer(train_texts, val_texts):
        tokenizer = text.Tokenizer(num_words=TOP_K)
        tokenizer.fit_on_texts(train_texts)

        X_train = tokenizer.texts_to_sequences(train_texts)
        X_val = tokenizer.texts_to_sequences(val_texts)

        # The Google tutorials have sequence padding going here, but this is unnecessary as the dataset has already been curtailed to 1000 tokens per line of the .csv file.

        return X_train, X_val, tokenizer.word_index

