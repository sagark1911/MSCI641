#!/usr/bin/env python
# coding: utf-8

# In[29]:


import sys
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, naive_bayes, metrics
from hypopt import GridSearch
import ast
import numpy as np


# In[30]:


f = open('Dataset/pos/train.csv','r')

training_pos = []

for line in f.readlines():
    training_pos.append(ast.literal_eval(line))


# In[31]:


f2 = open('Dataset/neg/train.csv','r')

training_neg = []

for line in f2.readlines():
    training_neg.append(ast.literal_eval(line)) 


# In[32]:


f3 = open('Dataset/pos/val.csv','r')
val_pos = []

for line in f3.readlines():
    val_pos.append(ast.literal_eval(line)) 


# In[33]:


f4 = open('Dataset/neg/val.csv','r')
val_neg = []

for line in f4.readlines():
    val_neg.append(ast.literal_eval(line)) 


# In[34]:


f5 = open('Dataset/pos/test.csv','r')
test_pos = []

for line in f5.readlines():
    test_pos.append(ast.literal_eval(line)) 


# In[35]:


f6 = open('Dataset/neg/test.csv','r')
test_neg = []

for line in f6.readlines():
    test_neg.append(ast.literal_eval(line)) 


# In[36]:


#training_data
vectorizer = CountVectorizer(tokenizer=lambda text: text, lowercase=False)
train = vectorizer.fit_transform(training_pos + training_neg)


# In[38]:


#validation_data
validation = vectorizer.transform(val_pos + val_neg)


# In[39]:


test = vectorizer.transform(test_pos + test_neg)


# In[40]:


train_label = np.concatenate((np.ones(len(training_pos)),np.zeros(len(training_neg))))
validation_label = np.concatenate((np.ones(len(val_pos)), np.zeros(len(val_neg))))
test_label = np.concatenate((np.ones(len(test_pos)), np.zeros(len(test_neg))))


# In[41]:


opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [1, 2, 3]}])
opt.fit(train, train_label, validation, validation_label)
print('Test Score for Optimized Parameters: {:.4f}'.format(opt.score(test, test_label)))
    


# In[ ]:




