from sklearn.datasets import load_iris
iris = load_iris()
examples = iris.data
truths = iris.target

from sklearn.model_selection import train_test_split
train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_examples, train_truths)
prediction = decision_tree.predict(test_examples)

from sklearn.metrics import accuracy_score
print("result with decision tree:", accuracy_score(test_truths, prediction))

from sklearn.tree import export_graphviz
export_graphviz(decision_tree, out_file='decision_tree_iris.dot')
