import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import urllib.request
from io import StringIO, BytesIO, TextIOWrapper
from zipfile import ZipFile

uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00454/"
victorian_authorship_file = "dataset.zip"

response = urllib.request.urlopen(uci_url + urllib.request.quote(victorian_authorship_file))

zipfile = ZipFile(BytesIO(response.read()))

data = TextIOWrapper(zipfile.open('dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv'))

df = pd.read_csv(data)
#print(df.head(25))

X = df.text
y = df.author
#print(X.head())
#tfidf = TfidfVectorizer(min_df=5,
                        #ngram_range=(1, 2), 
                        #stop_words='english')

vectorizer = CountVectorizer(max_features=2000)
BOW = vectorizer.fit_transform(X)

#features = tfidf.fit_transform(df.text)
#labels = df.author

X_train, X_test, y_train, y_test = train_test_split(BOW, np.asarray(y), train_size=0.8, random_state = 0)

linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)

y_pred = linear.predict(X_train)
print(accuracy_score(y, y_pred))

#rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
#poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
#sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)
#
#h = .01
##create the mesh
#x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
## create the title that will be shown on the plot
#titles = ['Linear kernel','RBF kernel','Polynomial kernel','Sigmoid kernel']
#
#for i, clf in enumerate((linear, rbf, poly, sig)):
    ##defines how many plots: 2 rows, 2columns=> leading to 4 plots
    #plt.subplot(2, 2, i + 1) #i+1 is the index
    ##space between plots
    #plt.subplots_adjust(wspace=0.4, hspace=0.4) 
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    ## Put the result into a color plot
    #Z = Z.reshape(xx.shape)
    #plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
    ## Plot also the training points
    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn,     edgecolors='grey')
    #plt.xlabel('Sepal length')
    #plt.ylabel('Sepal width')
    #plt.xlim(xx.min(), xx.max())
    #plt.ylim(yy.min(), yy.max())
    #plt.xticks(())
    #plt.yticks(())
    #plt.title(titles[i])
    #plt.show()