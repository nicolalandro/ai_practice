import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
names_without_quality = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                         'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
dataset = pd.read_csv('data/winequality-red.csv', names=names)

examples = dataset.drop(['quality'], axis=1).values
truths = dataset['quality'].get_values()

train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_examples, train_truths)
prediction = classifier.predict(test_examples)
print("knn result:", accuracy_score(test_truths, prediction))

classifier = SVC()
classifier.fit(train_examples, train_truths)
prediction = classifier.predict(test_examples)
print("svn result:", accuracy_score(test_truths, prediction))

