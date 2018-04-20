import pandas as pd
from sklearn.model_selection import train_test_split

names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
dataset = pd.read_csv('data/winequality-red.csv', names=names)

examples = dataset.drop(['quality'], axis=1).values
truths = dataset['quality'].get_values()

train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

# from sklearn.metrics import accuracy_score
# print("knn result:", accuracy_score(test_truths, prediction))
