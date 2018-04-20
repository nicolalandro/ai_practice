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

from sklearn.cluster import KMeans
knn = KMeans(n_clusters=4)
knn.fit(x_train)
prediction = knn.predict(x_test)

print(prediction)
print(y_test)

new_examples = []

for i in y_test:
	if(i == 2):
		new_examples.append(2)
	elif(i == 1):
		new_examples.append(0)
	else:
		new_examples.append(1)

#for i in range(0, examples.size - 1):
#	print(knn.predict([examples[i]]), ' - ', truths[i])

from sklearn.metrics import accuracy_score
print("ciao", accuracy_score(new_examples, prediction))
