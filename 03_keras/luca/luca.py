from keras.models import Sequential
import numpy as np
import pandas as pd
from keras.layers import Dense
from sklearn.model_selection import train_test_split

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=1599))
model.add(Dense(units=10, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))

names_without_quality = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
dataset = pd.read_csv('../data/winequality-red.csv')

x_train = []
y_train = []

for index, row in dataset.iterrows():
    y_train.append(row[-1])
    # print(list(row))
    # row.drop(row[:-1])
    # print(list(row))
    x_train.append(row[:-1])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33)

# x_train = np.array([[1.0, 2.3], [2.0, 3.3]])
# y_train = np.array([0.99, 2.2])
# print(x_train)
# print(y_train)
print(x_train.size())
model.fit(np.array(x_train), np.array(y_train), epochs=5, batch_size=32)

# loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
#
# print(loss_and_metrics)