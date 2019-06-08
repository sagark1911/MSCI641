#!/usr/bin/env python
# coding: utf-8
# Author : Sagar Kulkarni


import sys
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, naive_bayes, metrics
from hypopt import GridSearch
import ast
import numpy as np


# In[24]:
if __name__ =='__main__':

	f = open(sys.argv[1],'r')

	training_pos = []

	for line in f.readlines():
	    training_pos.append(ast.literal_eval(line))


# In[25]:


	f2 = open(sys.argv[2],'r')

	training_neg = []

	for line in f2.readlines():
	    training_neg.append(ast.literal_eval(line)) 


# In[26]:


	f3 = open(sys.argv[3],'r')
	val_pos = []

	for line in f3.readlines():
	    val_pos.append(ast.literal_eval(line)) 


	# In[27]:


	f4 = open(sys.argv[4],'r')
	val_neg = []

	for line in f4.readlines():
	    val_neg.append(ast.literal_eval(line)) 


	# In[28]:


	f5 = open(sys.argv[5],'r')
	test_pos = []

	for line in f5.readlines():
	    test_pos.append(ast.literal_eval(line)) 


	# In[29]:


	f6 = open(sys.argv[6],'r')
	test_neg = []

	for line in f6.readlines():
	    test_neg.append(ast.literal_eval(line)) 


	# In[30]:


#	training_pos = training_pos[0:20000]
#	training_neg = training_neg[0:20000]


	# In[35]:


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
	mnb = naive_bayes.MultinomialNB(alpha = 0)
	mnb.fit(train,train_label)
	y_pred = mnb.predict(validation)
	print("Unigrams Validation Accuracy Alpha = 0 :",metrics.accuracy_score(validation_label, y_pred))


	# In[36]:


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
	mnb = naive_bayes.MultinomialNB(alpha = 1)
	mnb.fit(train,train_label)
	y_pred = mnb.predict(validation)
	print("Unigrams Validation Accuracy Alpha = 1 :",metrics.accuracy_score(validation_label, y_pred))


	# In[37]:


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
	mnb = naive_bayes.MultinomialNB(alpha = 3)
	mnb.fit(train,train_label)
	y_pred = mnb.predict(validation)
	print("Unigrams Validation Accuracy Alpha = 3:",metrics.accuracy_score(validation_label, y_pred))


	# In[34]:


	opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,1,2,3,4]}])
	opt.fit(train, train_label, validation, validation_label)
	print('Test Score for Optimized Parameters: {:.4f}'.format(opt.score(test, test_label)))


	# In[39]:


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
	mnb = naive_bayes.MultinomialNB(alpha = 0)
	mnb.fit(train,train_label)
	y_pred = mnb.predict(validation)
	print("Bigrams Validation Accuracy Alpha = 0:",metrics.accuracy_score(validation_label, y_pred))


	# In[40]:


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
	mnb = naive_bayes.MultinomialNB(alpha = 1)
	mnb.fit(train,train_label)
	y_pred = mnb.predict(validation)
	print("Bigrams Validation Accuracy Alpha = 1:",metrics.accuracy_score(validation_label, y_pred))


	# In[41]:


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
	mnb = naive_bayes.MultinomialNB(alpha = 3)
	mnb.fit(train,train_label)
	y_pred = mnb.predict(validation)
	print("Bigrams Validation Accuracy Alpha = 3:",metrics.accuracy_score(validation_label, y_pred))


	# In[42]:


	opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,1,2,3,4]}])
	opt.fit(train, train_label, validation, validation_label)
	print('Test Score for Optimized Parameters: {:.4f}'.format(opt.score(test, test_label)))


	# In[43]:


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
	mnb = naive_bayes.MultinomialNB(alpha = 0)
	mnb.fit(train,train_label)
	y_pred = mnb.predict(validation)
	print("Unigrams + Bigrams Validation Accuracy Alpha = 0:",metrics.accuracy_score(validation_label, y_pred))


	# In[44]:


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
	mnb = naive_bayes.MultinomialNB(alpha = 1)
	mnb.fit(train,train_label)
	y_pred = mnb.predict(validation)
	print("Unigrams + Bigrams Validation Accuracy Alpha = 1:",metrics.accuracy_score(validation_label, y_pred))


	# In[45]:


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
	mnb = naive_bayes.MultinomialNB(alpha = 2)
	mnb.fit(train,train_label)
	y_pred = mnb.predict(validation)
	print("Unigrams + Bigrams Validation Accuracy Alpha = 2:",metrics.accuracy_score(validation_label, y_pred))


	# In[46]:


	opt = GridSearch(model = naive_bayes.MultinomialNB(), param_grid = [{'alpha': [0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,1,2,3,4]}])
	opt.fit(train, train_label, validation, validation_label)
	print('Test Score for Optimized Parameters: {:.4f}'.format(opt.score(test, test_label)))


# In[ ]:




