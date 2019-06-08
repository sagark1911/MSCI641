from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import ast
from collections import Counter
from sklearn import metrics


f = open('Dataset/pos/train.csv','r')

training_pos = []

for line in f.readlines():
    training_pos.append(ast.literal_eval(line))

f2 = open('Dataset/neg/train.csv','r')

training_neg = []

for line in f2.readlines():
    training_neg.append(ast.literal_eval(line)) 


f3 = open('Dataset/pos/val.csv','r')
val_pos = []

for line in f3.readlines():
    val_pos.append(ast.literal_eval(line)) 


f4 = open('Dataset/neg/val.csv','r')
val_neg = []

for line in f4.readlines():
    val_neg.append(ast.literal_eval(line)) 


###########
training_pos = training_pos[0:10000]
training_neg = training_neg[0:10000]
###########



pos_bigrams = []
for line in training_pos:
    bigrams = []
    for i in range(0,len(line)-1):
        bi = line[i] + " " + line[i+1]
        bigrams.append(bi)
    pos_bigrams.append(bigrams)


neg_bigrams = []
for line in training_neg:
    bigrams = []
    for i in range(0,len(line)-1):
        bi = line[i] + " " + line[i+1]
        bigrams.append(bi)
    neg_bigrams.append(bigrams)


val_pos_bigrams = []
for line in val_pos:
    bigrams = []
    for i in range(0,len(line)-1):
        bi = line[i] + " " + line[i+1]
        bigrams.append(bi)
    val_pos_bigrams.append(bigrams)

val_neg_bigrams = []
for line in val_neg:
    bigrams = []
    for i in range(0,len(line)-1):
        bi = line[i] + " " + line[i+1]
        bigrams.append(bi)
    val_neg_bigrams.append(bigrams)



pos_train_data = []
for training_data in training_pos:
    pos_train_data.append(dict(Counter(training_data)))


neg_train_data = []
for training_data in training_neg:
    neg_train_data.append(dict(Counter(training_data)))





val_pos_data = []
for val_data in val_pos:
    val_pos_data.append(dict(Counter(val_data)))

###DELETE BEFORE SUBMISSION
val_pos_data = val_pos_data
###DELETEBEFORE SUBMISSION

pos_val_label = np.ones(len(val_pos_data))

val_neg_data = []
for val_data in val_neg:
    val_neg_data.append(dict(Counter(val_data)))
###DELETE BEFORE SUBMISSION
val_neg_data = val_neg_data 
###DELETEBEFORE SUBMISSION

neg_val_label = np.zeros(len(val_neg_data))

val_data =  val_neg_data + val_pos_data 

val_label = np.concatenate((neg_val_label,pos_val_label))



pos_train_data_bigrams = []
for training_data in pos_bigrams:
    pos_train_data_bigrams.append(dict(Counter(training_data)))


# In[9]:

    
neg_train_data_bigrams = []
for training_data in neg_bigrams:
    d = dict()
    for dat in training_data:
        if dat in d:
            d[dat] = d[dat] + 1
        else:
            d[dat] = 1
    neg_train_data_bigrams.append(d)



val_pos_data_bigrams = []
for val_data in val_pos_bigrams:
    val_pos_data_bigrams.append(dict(Counter(val_data)))

###DELETE BEFORE SUBMISSION
val_pos_data_bigrams = val_pos_data_bigrams
###DELETEBEFORE SUBMISSION

pos_val_label_bigrams = np.ones(len(val_pos_data_bigrams))

val_neg_data_bigrams = []
for val_data in val_neg_bigrams:
    val_neg_data_bigrams.append(dict(Counter(val_data)))


###DELETE BEFORE SUBMISSION
val_neg_data_bigrams = val_neg_data_bigrams 
###DELETEBEFORE SUBMISSION

neg_val_label_bigrams = np.zeros(len(val_neg_data_bigrams))

val_data_bigrams =  val_neg_data_bigrams + val_pos_data_bigrams 

val_label_bigrams = np.concatenate((neg_val_label_bigrams,pos_val_label_bigrams))









####TO DELETE BEFORE SUBMITTING
neg_train_data = neg_train_data[0:5000]
pos_train_data = pos_train_data[0:5000]
####TO DELETE BEFORE SUBMITTING END //


train_data = neg_train_data + pos_train_data
pos_label = np.ones(len(pos_train_data))
neg_label = np.zeros(len(neg_train_data))
label = np.concatenate((neg_label,pos_label))
label


# In[12]:


dv = DictVectorizer(sparse=False)

dv.fit(train_data)

X = []
Y = np.array(label)

mnb = MultinomialNB()

training_length = min(len(neg_train_data),len(pos_train_data))

for i in range(training_length):
    train = []
    train.append(neg_train_data[i])
    train.append(pos_train_data[i])
    X = dv.transform(train)
    Y1 = [0,1]
    mnb.partial_fit(X,Y1, classes = np.unique(Y1))
    if i%100 == 0:
        y_pred = []
        for i in range(len(val_data)):
            val = dv.transform(val_data[i])
            y_pred.append(mnb.predict(val))

        print("Validation Accuracy:",metrics.accuracy_score(val_label, y_pred))

    



# In[ ]:

#JUST bigrams
####TO DELETE BEFORE SUBMITTING
neg_train_data_bigrams = neg_train_data_bigrams[0:5000]
pos_train_data_bigrams = pos_train_data_bigrams[0:5000]
####TO DELETE BEFORE SUBMITTING END //


train_data = neg_train_data_bigrams + pos_train_data_bigrams
pos_label = np.ones(len(pos_train_data_bigrams))
neg_label = np.zeros(len(neg_train_data_bigrams))
label = np.concatenate((neg_label,pos_label))

dv = DictVectorizer(sparse=False)

dv.fit(train_data)

X = []
Y = np.array(label)

mnb = MultinomialNB()

training_length = min(len(neg_train_data),len(pos_train_data))

print("Training with bigrams and val on bigrams")

for i in range(training_length):
    train = []
    train.append(neg_train_data[i])
    train.append(pos_train_data[i])
    X = dv.transform(train)
    Y1 = [0,1]
    mnb.partial_fit(X,Y1, classes = np.unique(Y1))
    if i%100 == 0:
        y_pred = []
        for i in range(len(val_data_bigrams)):
            val = dv.transform(val_data_bigrams[i])
            y_pred.append(mnb.predict(val))

        print("Validation Accuracy:",metrics.accuracy_score(val_label_bigrams, y_pred))

    







# In[ ]:


val_neg = ['these', 'were', 'okay', 'but', 'arrived', 'tasting', 'old', '.', '.', '.', 'we', 'refrigerated', 'them', 'and', 'they', 'got', 'sticky', '.', '.', 'were', 'afraid', 'they', 'were', "'", 't', "'", 'good', 'so', 'ended', 'up', 'throwing', 'them', 'away', '.']
val_pos = ['came', 'in', 'one', 'piece', ',', 'nicely', 'packed', 'and', 'works', 'perfect', 'for', 'my', 'rice', 'and', 'cous', 'cous', 'which', 'i', 'buy', 'in', 'bulk', 'and', 'store', 'on', 'my', 'counter', '.']


# In[ ]:


val = [dict(Counter(val_neg))]
VAL = dv.transform(val)

mnb.predict(VAL)


# In[ ]:


val = [dict(Counter(val_pos))]
VAL = dv.transform(val)

mnb.predict(VAL)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


validation_pos = pd.read_csv(folder + '/' + 'pos' + '/' + 'val.csv')
validation_neg = pd.read_csv(folder + '/' + 'neg' + '/' + 'val.csv')
test_pos = pd.read_csv(folder + '/' + 'pos' + '/' + 'test.csv')
test_neg = pd.read_csv(folder + '/' + 'neg' + '/' + 'test.csv')


# In[ ]:




