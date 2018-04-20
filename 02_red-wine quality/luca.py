import numpy as np
import pandas as pd

names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
names_without_quality = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                         'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
dataset = pd.read_csv('data/winequality-red.csv', names=names)

examples_by_quality = {}
truths_by_quality = []

for index, row in dataset.iterrows():
    example = []
    for name in names_without_quality:
        example.append(float(row[name]))
    if row['quality'] not in examples_by_quality.keys:
        examples_by_quality[row['quality']] = []

    examples_by_quality[row['quality']].append(example)

    # if truths_by_quality[row['quality']] is None:
    #     truths_by_quality[row['quality']] = []
    #
    # truths_by_quality[row['quality']]
for key, value in examples_by_quality:
    np.savetxt(key + '.txt', value.corr())