import keras
from sklearn.datasets import load_iris


def class_number_to_array(n):
    out = [0.0, 0.0, 0.0]
    out[n] = 1.0
    return out


iris = load_iris()
examples = iris.data
truths = iris.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(examples, truths, test_size=0.33)

from keras.models import Sequential

model = Sequential()
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=4))
model.add(Dense(units=64, activation='sigmoid', input_dim=4))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(x_train, y_train, epochs=4, batch_size=32)

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print(loss_and_metrics)

prediction = model.predict(x_test, batch_size=128)
print(prediction)
# import numpy as np
# test_truth = np.asarray(map(class_number_to_array, y_test))
# print(type(test_truth))
# print(type(prediction))