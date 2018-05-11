# https://www.kaggle.com/carlosaguayo/deep-learning-for-text-classification/code

import os
import numpy as np
from keras.layers import Activation, Conv1D, Dense, Embedding, Flatten, Input, MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.base import get_data_home

dataset = fetch_20newsgroups(subset='all', shuffle=True, download_if_missing=False)

texts = dataset.data
target = dataset.target

# print(len(texts[0].split()))
print(texts[0])
print(target[0])
print(dataset.target_names[target[0]])
