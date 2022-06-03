import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/spicy.kev/Desktop/victorian_authorship/Gungor_2018_VictorianAuthorAttribution_data-train.csv', encoding='latin-1')

print(df.dtypes)

X = df.text
y = df.author

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)

X_test_dtm = vect.transform(X_test)

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)
print(metrics.accuracy_score(y_test, y_pred_class))

print(y_test.value_counts())
null_accuracy = y_test.value_counts().head(1)/len(y_test)
print('Null accuracy: ', null_accuracy)

print(metrics.confusion_matrix(y_test, y_pred_class))

#print(X_test[y_pred_class > y_test])

y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train_dtm, y_train)

y_pred_class = logreg.predict(X_test_dtm)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
print(metrics.accuracy_score(y_test, y_pred_class))




