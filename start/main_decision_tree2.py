examples = [[36,0,0],[0,32,32],[35,1,0],[34,0,0]]
truths= [0,1,0,0]

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(examples, truths)
#prediction = decision_tree.predict(test_examples)

from sklearn.tree import export_graphviz
export_graphviz(decision_tree, out_file='decision_tree_iris.dot')
