import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

df = pd.read_csv(r'data\Gungor_2018_VictorianAuthorAttribution_data-train.csv', encoding='latin-1')
print(df.head())

print(df.author.unique())
df['author'] = df['author'].astype(str).replace()

EMBEDDING_DIMENSION = 64
VOCABULARY_SIZE = 2000
MAX_LENGTH = 100
OOV_TOK = '<OOV>'
TRUNCATE_TYPE = 'post'
PADDING_TYPE ='post'

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['author'], test_size=0.33, random_state=55)

print(X_train.shape, X_test.shape, y_train)

tokenizer = Tokenizer(num_words = VOCABULARY_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)
word_index = tokenizer.word_index
print("Vocabularuy Size: ", len(word_index))
dict(list(word_index.items())[0:10])

#print(X_train_sequences[100])
#
X_train_pad = sequence.pad_sequences(X_train_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATE_TYPE)
X_test_pad = sequence.pad_sequences(X_test_sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATE_TYPE)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(list(y_train))

training_label_seq = np.array(label_tokenizer.texts_to_sequences(y_train))
test_label_seq = np.array(label_tokenizer.texts_to_sequences(y_test))

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_text(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     EMBEDDING_DIMENSION))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(EMBEDDING_DIMENSION, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(50))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

num_epochs = 10
history = model.fit(X_train_pad, training_label_seq, epochs=num_epochs, validation_data=(X_test_pad, test_label_seq), verbose=2)

def graph_plots(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

graph_plots(history, 'accuracy')
graph_plots(history, 'loss')

ypred = model.predict(X_train, verbose=1)
ypred = np.argmax(ypred, axis=1)

ypred = model.predict(X_train, verbose=1)
ypred = np.argmax(ypred, axis=1)

target_names = df['author']

print(classification_report(np.argmax(y_train, axis=1), ypred, target_names=target_names))

cm = confusion_matrix(np.argmax(y_train, axis=1), ypred)
cm = pd.DataFrame(cm, range(2),range(2))
plt.figure(figsize = (10,10))

sns.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()
