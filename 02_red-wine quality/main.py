from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd

names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
names_without_quality = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
dataset = pd.read_csv('data/winequality-red.csv', names=names)

examples = []
truths = []
for index, row in dataset.iterrows():
    example = []
    for name in names_without_quality:
        example.append(float(row[name]))
    examples.append(example)
    truths.append(row['quality'])


train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_examples, train_truths)
prediction = classifier.predict(test_examples)
# print("result:", accuracy_score(test_truths, prediction))
corrcoef = np.corrcoef(examples)
cov_matrix = np.cov(examples)

# uniques = []
# for row in corrcoef:
#     if np.amin(row) <= 0.6:
#         print(list(row).index(np.amin(row)))
#         uniques.append(list(row).index(np.amin(row)))
#
# print set(uniques)