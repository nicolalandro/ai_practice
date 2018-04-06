import numpy as np
import pandas as pd

names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
names_without_quality = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                         'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
dataset = pd.read_csv('data/winequality-red.csv', names=names)
np.savetxt('test.txt', dataset.corr())

class_3 = dataset[dataset['quality'] == 3]
class_4 = dataset[dataset['quality'] == 4]
class_5 = dataset[dataset['quality'] == 5]
class_6 = dataset[dataset['quality'] == 6]
class_7 = dataset[dataset['quality'] == 7]
class_8 = dataset[dataset['quality'] == 8]

np.savetxt('class3.txt', class_3.corr())
np.savetxt('class4.txt', class_4.corr())
np.savetxt('class5.txt', class_5.corr())
np.savetxt('class6.txt', class_6.corr())
np.savetxt('class7.txt', class_7.corr())
np.savetxt('class8.txt', class_8.corr())
