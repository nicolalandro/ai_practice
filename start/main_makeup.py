import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

examples = []
truths = []

dataset = pd.read_csv('makeup_dataset.csv', sep='|')
for index, row in dataset.iterrows():
	examples.append(str(row[1]) + ' ' + str(row[2]))
	truths.append(str(row[4]))

dictionary = CountVectorizer().fit(examples)
examples = dictionary.transform(examples)

from sklearn.model_selection import train_test_split
train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_examples, train_truths)
prediction = decision_tree.predict(test_examples)

from sklearn.metrics import accuracy_score
print("result with decision tree:", accuracy_score(test_truths, prediction))

test = 'ombretto blush cipria rossetto'
transformed_pattern = dictionary.transform([test])
print(decision_tree.predict(transformed_pattern))


from sklearn.neighbors import KNeighborsClassifier
decision_tree = KNeighborsClassifier()
decision_tree.fit(train_examples, train_truths)
prediction = decision_tree.predict(test_examples)

#from sklearn.metrics import accuracy_score
print("result with decision tree:", accuracy_score(test_truths, prediction))

test = 'ombretto blush cipria rossetto'
transformed_pattern = dictionary.transform([test])
print(decision_tree.predict(transformed_pattern))


from sklearn.ensemble import ExtraTreesClassifier
decision_tree = ExtraTreesClassifier(n_estimators=100, n_jobs=12, bootstrap=False, min_samples_split=2, random_state=0)
decision_tree.fit(train_examples, train_truths)
prediction = decision_tree.predict(test_examples)

#from sklearn.metrics import accuracy_score
print("result with decision tree:", accuracy_score(test_truths, prediction))

test = 'ombretto blush cipria rossetto'
transformed_pattern = dictionary.transform([test])
print(decision_tree.predict(transformed_pattern))