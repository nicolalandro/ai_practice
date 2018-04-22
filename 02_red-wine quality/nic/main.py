import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
dataset = pd.read_csv('data/winequality-red.csv', names=names)

examples = dataset.drop(['quality'], axis=1).values
truths = dataset['quality'].get_values()

train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_examples, train_truths)
prediction = classifier.predict(test_examples)
print("knn result:", accuracy_score(test_truths, prediction))

from sklearn.svm import SVC

classifier = SVC()
classifier.fit(train_examples, train_truths)
prediction = classifier.predict(test_examples)
print("svc result:", accuracy_score(test_truths, prediction))

from sklearn.svm import LinearSVC

classifier = LinearSVC()
classifier.fit(train_examples, train_truths)
prediction = classifier.predict(test_examples)
print("linear svc result:", accuracy_score(test_truths, prediction))

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(train_examples, train_truths)
prediction = classifier.predict(test_examples)
print("decision tree result:", accuracy_score(test_truths, prediction))

# use mean to train
print('<---------------------------------- addestro con la media ---------------------------------->')
class_3 = dataset[dataset['quality'] == 3].drop(['quality'], axis=1)
class_4 = dataset[dataset['quality'] == 4].drop(['quality'], axis=1)
class_5 = dataset[dataset['quality'] == 5].drop(['quality'], axis=1)
class_6 = dataset[dataset['quality'] == 6].drop(['quality'], axis=1)
class_7 = dataset[dataset['quality'] == 7].drop(['quality'], axis=1)
class_8 = dataset[dataset['quality'] == 8].drop(['quality'], axis=1)

mean_class3 = np.mean(class_3).values
mean_class4 = np.mean(class_4).values
mean_class5 = np.mean(class_5).values
mean_class6 = np.mean(class_6).values
mean_class7 = np.mean(class_7).values
mean_class8 = np.mean(class_8).values

truths_mean = [3, 4, 5, 6, 7, 8]
examples_mean = [mean_class3, mean_class4, mean_class5, mean_class6, mean_class7, mean_class8]

classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(examples_mean, truths_mean)
prediction = classifier.predict(test_examples)
print("knn from mean result:", accuracy_score(test_truths, prediction))

classifier = SVC()
classifier.fit(examples_mean, truths_mean)
prediction = classifier.predict(test_examples)
print("svc from mean result:", accuracy_score(test_truths, prediction))

classifier = LinearSVC()
classifier.fit(examples_mean, truths_mean)
prediction = classifier.predict(test_examples)
print("linear svc from mean result:", accuracy_score(test_truths, prediction))

classifier = DecisionTreeClassifier()
classifier.fit(examples_mean, truths_mean)
prediction = classifier.predict(test_examples)
print("decision tree from mean result:", accuracy_score(test_truths, prediction))