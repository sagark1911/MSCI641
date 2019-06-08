#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, naive_bayes, metrics
from hypopt import GridSearch
import ast
import numpy as np


# In[15]:


f = open('Dataset/pos/train.csv','r')

training_pos = []

for line in f.readlines():
    training_pos.append(ast.literal_eval(line))


# In[3]:


f2 = open('Dataset/neg/train.csv','r')

training_neg = []

for line in f2.readlines():
    training_neg.append(ast.literal_eval(line)) 


# In[4]:


f3 = open('Dataset/pos/val.csv','r')
val_pos = []

for line in f3.readlines():
    val_pos.append(ast.literal_eval(line)) 


# In[5]:


f4 = open('Dataset/neg/val.csv','r')
val_neg = []

for line in f4.readlines():
    val_neg.append(ast.literal_eval(line)) 


# In[6]:


f5 = open('Dataset/pos/test.csv','r')
test_pos = []

for line in f5.readlines():
    test_pos.append(ast.literal_eval(line)) 


# In[7]:


f6 = open('Dataset/neg/test.csv','r')
test_neg = []

for line in f6.readlines():
    test_neg.append(ast.literal_eval(line)) 


# In[13]:


training_pos = training_pos
training_neg = training_neg


# In[18]:


f = open('Dataset/pos/train.csv','r')
f2 = open('Dataset/neg/train.csv','r')


# In[ ]:





# In[19]:


#training_data
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))
vectorizer.fit(f.readlines() + f2.readlines())
train = vectorizer.transform(f.readlines() + f2.readlines())


# In[ ]:


#validation_data
validation = vectorizer.transform(val_pos + val_neg)


# In[ ]:


test = vectorizer.transform(test_pos + test_neg)


# In[ ]:


train_label = np.concatenate((np.ones(len(training_pos)),np.zeros(len(training_neg))))
validation_label = np.concatenate((np.ones(len(val_pos)), np.zeros(len(val_neg))))
test_label = np.concatenate((np.ones(len(test_pos)), np.zeros(len(test_neg))))


# In[ ]:


mnb = naive_bayes.MultinomialNB()
mnb.fit(train,train_label)
y_pred = mnb.predict(validation)
print("Validation Accuracy:",metrics.accuracy_score(validation_label, y_pred))


# In[ ]:




