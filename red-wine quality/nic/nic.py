import numpy as np
import pandas as pd

names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
names_without_quality = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                         'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
dataset = pd.read_csv('data/winequality-red.csv', names=names)
np.savetxt('nic/test.txt', dataset.corr())

class_3 = dataset[dataset['quality'] == 3]
class_4 = dataset[dataset['quality'] == 4]
class_5 = dataset[dataset['quality'] == 5]
class_6 = dataset[dataset['quality'] == 6]
class_7 = dataset[dataset['quality'] == 7]
class_8 = dataset[dataset['quality'] == 8]

np.savetxt('nic/class3.txt', class_3.corr())
np.savetxt('nic/class4.txt', class_4.corr())
np.savetxt('nic/class5.txt', class_5.corr())
np.savetxt('nic/class6.txt', class_6.corr())
np.savetxt('nic/class7.txt', class_7.corr())
np.savetxt('nic/class8.txt', class_8.corr())

np.savetxt('nic/corr_coef3.txt', np.corrcoef(class_3))
np.savetxt('nic/corr_coef4.txt', np.corrcoef(class_4))
np.savetxt('nic/corr_coef5.txt', np.corrcoef(class_5))
np.savetxt('nic/corr_coef6.txt', np.corrcoef(class_6))
np.savetxt('nic/corr_coef7.txt', np.corrcoef(class_7))
np.savetxt('nic/corr_coef8.txt', np.corrcoef(class_8))

mean_class3 = np.mean(class_3)
mean_class4 = np.mean(class_4)
mean_class5 = np.mean(class_5)
mean_class6 = np.mean(class_6)
mean_class7 = np.mean(class_7)
mean_class8 = np.mean(class_8)

np.savetxt('nic/mean_3.txt', mean_class3)
np.savetxt('nic/mean_4.txt', mean_class4)
np.savetxt('nic/mean_5.txt', mean_class5)
np.savetxt('nic/mean_6.txt', mean_class6)
np.savetxt('nic/mean_7.txt', mean_class7)
np.savetxt('nic/mean_8.txt', mean_class8)

np.savetxt('nic/corr_media_diff.txt',
           np.corrcoef([mean_class3, mean_class4, mean_class5, mean_class6, mean_class7, mean_class8]))

# Le medie si assomigliano troppo

exponential_transponse = np.power([mean_class3, mean_class4, mean_class5, mean_class6, mean_class7, mean_class8], 5)
np.savetxt('nic/corr_media_exp_diff.txt',
           np.corrcoef(exponential_transponse))

