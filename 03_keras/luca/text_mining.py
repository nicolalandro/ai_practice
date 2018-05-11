from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('multinomial NB', MultinomialNB())
])

clf.fit(twenty_train.data, twenty_train.target)

predicted = clf.predict(twenty_test.data)
# print(np.mean(predicted == twenty_test.target))
print(accuracy_score(twenty_test.target, predicted))

clf2 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegressionWithLBFGS())
])

clf2.fit(twenty_train.data, twenty_train.target)

predicted2 = clf2.predict(twenty_test.data)
print(accuracy_score(twenty_test.target, predicted2))
