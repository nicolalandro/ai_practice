from sklearn.datasets import load_iris
iris = load_iris()
examples = iris.data
truths = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(examples, truths, test_size=0.33)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
prediction = knn.predict(examples)

print(prediction)
print(y_test)

#from sklearn.metrics import accuracy_score
#print("ciao", accuracy_score(y_test, prediction))
