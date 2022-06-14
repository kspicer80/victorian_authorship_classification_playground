import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns
import urllib.request
from io import StringIO, BytesIO, TextIOWrapper
from zipfile import ZipFile

uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00454/"
victorian_authorship_file = "dataset.zip"

response = urllib.request.urlopen(uci_url + urllib.request.quote(victorian_authorship_file))

zipfile = ZipFile(BytesIO(response.read()))

data = TextIOWrapper(zipfile.open('dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv'))

df = pd.read_csv(data, encoding='latin-1', low_memory=False)
#X_sample = df.sample(frac=.25, replace=False, random_state=1)
X = df.text
y = df.author

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=20, ngram_range=(1,2), stop_words='english')

features = tfidf.fit_transform(X)

labels = df.author

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
    std_accuracy = cv_df.groupby('model_name').accuracy.std()
    acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, ignore_index=True)

    acc.columns = ['Mean Accuracy', 'Standard deviation']
    print(acc)
'''
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.20, random_state=1)

svc_model = LinearSVC()
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)

print("Classification Report Metrics:\n")
print(metrics.classification_report(y_test, y_pred))

confusion_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix for Linear SCV")
plt.show()

rf = RandomForestClassifier(n_estimators = 150, max_depth=10, random_state=1)
print(rf.get_params())
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(rf.score(X_test, y_test).round(4))
conf_mat = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix for Linear SCV")
plt.show()
'''

