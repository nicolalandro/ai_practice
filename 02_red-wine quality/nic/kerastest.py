import numpy as np
import pandas as pd
from keras.backend import categorical_crossentropy
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn


def class_number_to_array(n):
    out = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    out[n] = 1.0
    return out


names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
dataset = pd.read_csv('data/winequality-red.csv', names=names)

examples = dataset.drop(['quality'], axis=1).values
truths = dataset['quality'].get_values()

sns_plot = seaborn.countplot(truths)
sns_plot.figure.savefig("keras_info/rating_tot.png")
plt.close()

train_examples, test_examples, train_truths, test_truths = train_test_split(examples, truths, test_size=0.33)

sns_plot = seaborn.countplot(train_truths)
sns_plot.figure.savefig("keras_info/rating_train.png")
plt.close()

sns_plot = seaborn.countplot(test_truths)
sns_plot.figure.savefig("keras_info/rating_test.png")
plt.close()

train_truths = np.array(list(map(class_number_to_array, train_truths)))
test_truths = np.array(list(map(class_number_to_array, test_truths)))

model = Sequential()
model.add(Dense(units=22, activation='relu', input_dim=11))
model.add(Dense(units=45, activation='sigmoid', input_dim=11))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
model.fit(train_examples, train_truths, epochs=10, batch_size=32)

loss_and_metrics = model.evaluate(test_examples, test_truths, batch_size=128)
print("loss and metrics: ", loss_and_metrics)

prediction = model.predict(test_examples, batch_size=128)
prediction = prediction.argmax(1)
test_truths = test_truths.argmax(1)

print("accuracy score: ", accuracy_score(test_truths, prediction))
print("decision tree recal macro: ", recall_score(test_truths, prediction, average='macro'))
print("decision tree recal micro: ", recall_score(test_truths, prediction, average='micro'))
print("decision tree recal weighted: ", recall_score(test_truths, prediction, average='weighted'))
print("decision tree recal none: ", recall_score(test_truths, prediction, average=None))

cnf_matrix = confusion_matrix(test_truths, prediction)
sns_plot = seaborn.heatmap(cnf_matrix, annot=True, center=0)
sns_plot.figure.savefig("keras_info/cnf_matrix.png")
plt.close()


normalized_cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
sns_plot = seaborn.heatmap(normalized_cnf_matrix, annot=True, center=0)
sns_plot.figure.savefig("keras_info/normalized_cnf_matrix.png")
plt.close()
