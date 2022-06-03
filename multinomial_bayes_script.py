import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer

#df = pd.read_csv(r'data\Gungor_2018_VictorianAuthorAttribution_data-train.csv', encoding='latin-1')
#print(df.head())
#
#cv = CountVectorizer()
#X = cv.fit_transform(df.text)
#
#y = df.author
#
#clf = MultinomialNB()
#clf.fit(X, y)
#
#yhat = clf.predict(X)
#
#print("Accuracy: ", accuracy_score(y, yhat))

train = pd.read_csv(r'data\Gungor_2018_VictorianAuthorAttribution_data-train.csv', encoding='latin-1')
test = pd.read_csv(r'data\Gungor_2018_VictorianAuthorAttribution_data.csv', encoding='latin-1')

allData = pd.concat([train, test])

vec = TfidfVectorizer(ngram_range = (1,2))
vec.fit(allData.text)

train = allData.iloc[:train[0, :]] 
test = allData.iloc[train.shape[0]:, :]

trainMatrix = vec.transform(train.text)

def plot_learning_curve(trainSizes, trainScores, testScores, title="Learning Curve", filePath=None):
    fig, ax = plt.subplots(1,1, figsize=(9, 5))
    trainMeans =  np.mean(trainScores, axis=1)
    testMeans = np.mean(testScores, axis=1)

    ax.plot(trainSizes,trainMeans, color='blue', label='Training Scores')
    ax.plot(trainSizes, testMeans, color='red', label='Validation Scores')

    ax.set_title(title)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Number of examples')
    ax.legend(loc='best')
    if filePath:
        plt.savefig(filePath, bbox_inches='tight' )
        
from sklearn.naive_bayes import MultinomialNB 
from sklearn.learning_curve import learning_curve
trainSizes, NBTrainScores, NBTestScores = learning_curve(MultinomialNB(), trainMatrix, train.cuisine)
plot_learning_curve(trainSizes, NBTrainScores, NBTestScores, title='Multinomial Naive Bayes', filePath='mnb.png')

