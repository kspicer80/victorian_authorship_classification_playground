import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

file_path = '/Users/spicy.kev/Desktop/victorian_authorship/victorian_training_data.csv'

df = pd.read_csv(file_path, encoding='latin-1')

X = df.text
y = df.author
print(X.head())
tfidf = TfidfVectorizer(min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')


features = tfidf.fit_transform(df.text)
labels = df.author

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)

#random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
linear_SVC_model = LinearSVC()
#multinomial_nb_model = MultinomialNB
#logistic_regression)model = LogisticRegression(random_state=0)

CV = 5
accuracies = cross_val_score(linear_SVC_model, features, labels, scoring='accuracy', cv=CV)
print(accuracies)
print(f"The mean score is: {accuracies.mean()} and the standard deviation is: {accuracies.std()}.")
