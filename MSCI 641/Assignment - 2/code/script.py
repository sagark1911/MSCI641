from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import ast
from collections import Counter

f = open('Dataset/pos/train.csv','r')

training_pos = []

for line in f.readlines():
	training_pos.append(ast.literal_eval(line))

f2 = open('Dataset/neg/train.csv','r')

training_neg = []

for line in f2.readlines():
	training_neg.append(ast.literal_eval(line))
    
 
pos_train_data = []
for training_data in training_pos:
	pos_train_data.append(dict(Counter(training_data)))

neg_train_data = []
for training_data in training_neg:
	neg_train_data.append(dict(Counter(training_data)))

train_data = neg_train_data + pos_train_data
pos_label = np.ones(len(pos_train_data))
neg_label = np.zeros(len(neg_train_data))
label = np.concatenate((neg_label,pos_label))

dv = DictVectorizer(sparse=False)
dv.fit(train_data)
X = dv.transform(train_data)
Y = np.array(label)