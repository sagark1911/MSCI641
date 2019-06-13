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


# In[2]:


f = open(sys.argv[1])

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


# In[15]:


training_pos = training_pos[0:40000]
training_neg = training_neg[0:40000]


# In[16]:


#training_data only unigram
vectorizer = CountVectorizer(analyzer='word',tokenizer=lambda text: text, lowercase=False, ngram_range=(1,1))
vectorizer.fit(training_pos + training_neg)
train = vectorizer.transform(training_pos + training_neg)
#validation_data
validation = vectorizer.transform(val_pos + val_neg)
#test data
test = vectorizer.transform(test_pos + test_neg)
#labels
train_label = np.concatenate((np.ones(len(training_pos)),np.zeros(len(training_neg))))
validation_label = np.concatenate((np.ones(len(val_pos)), np.zeros(len(val_neg))))
test_label = np.concatenate((np.ones(len(test_pos)), np.zeros(len(test_neg))))
mnb = naive_bayes.MultinomialNB()
mnb.fit(train,train_label)
y_pred = mnb.predict(validation)
print("Unigrams Validation Accuracy:",metrics.accuracy_score(validation_label, y_pred))
print("Unigrams Test Accuracy:")

# In[17]:


opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,1,2,3,4]}])
opt.fit(train, train_label, validation, validation_label)
print('Test Score for Optimized Parameters: {:.4f}'.format(opt.score(test, test_label)))


# In[18]:


#training_data only bigram
vectorizer = CountVectorizer(analyzer='word',tokenizer=lambda text: text, lowercase=False, ngram_range=(2,2))
vectorizer.fit(training_pos + training_neg)
train = vectorizer.transform(training_pos + training_neg)
#validation_data
validation = vectorizer.transform(val_pos + val_neg)
#test data
test = vectorizer.transform(test_pos + test_neg)
#labels
train_label = np.concatenate((np.ones(len(training_pos)),np.zeros(len(training_neg))))
validation_label = np.concatenate((np.ones(len(val_pos)), np.zeros(len(val_neg))))
test_label = np.concatenate((np.ones(len(test_pos)), np.zeros(len(test_neg))))
mnb = naive_bayes.MultinomialNB()
mnb.fit(train,train_label)
y_pred = mnb.predict(validation)
print("Bigrams Validation Accuracy:",metrics.accuracy_score(validation_label, y_pred))


# In[19]:


opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,1,2,3,4]}])
opt.fit(train, train_label, validation, validation_label)
print('Test Score for Optimized Parameters: {:.4f}'.format(opt.score(test, test_label)))


# In[20]:


#training_data both bigram and unigram
vectorizer = CountVectorizer(analyzer='word',tokenizer=lambda text: text, lowercase=False, ngram_range=(1,2))
vectorizer.fit(training_pos + training_neg)
train = vectorizer.transform(training_pos + training_neg)
#validation_data
validation = vectorizer.transform(val_pos + val_neg)
#test data
test = vectorizer.transform(test_pos + test_neg)
#labels
train_label = np.concatenate((np.ones(len(training_pos)),np.zeros(len(training_neg))))
validation_label = np.concatenate((np.ones(len(val_pos)), np.zeros(len(val_neg))))
test_label = np.concatenate((np.ones(len(test_pos)), np.zeros(len(test_neg))))
mnb = naive_bayes.MultinomialNB()
mnb.fit(train,train_label)
y_pred = mnb.predict(validation)
print("Unigrams + Bigrams Validation Accuracy:",metrics.accuracy_score(validation_label, y_pred))


# In[21]:


opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,1,2,3,4]}])
opt.fit(train, train_label, validation, validation_label)
print('Test Score for Optimized Parameters: {:.4f}'.format(opt.score(test, test_label)))


# In[ ]:




