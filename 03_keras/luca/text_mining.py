from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
dictionary = count_vect.fit(twenty_train.data)
X_train_counts = dictionary.transform(twenty_train.data)
# X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
tfidf_clf = tfidf_transformer.fit(X_train_counts)
X_train_tfidf = tfidf_clf.transform(X_train_counts)
# X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
test_data = dictionary.transform(twenty_test.data)


predicted = clf.predict(tfidf_clf.transform(test_data))
print(np.mean(predicted == twenty_test.target))

from sklearn.metrics import accuracy_score
print(accuracy_score(twenty_test.target, predicted))