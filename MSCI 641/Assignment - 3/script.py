import sys
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
import numpy as np
import random
import multiprocessing
from gensim.models import Word2Vec
from time import time
import logging 
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


cores = multiprocessing.cpu_count() 

nltk.download("stopwords")


f_pos = open("Dataset/pos.txt","r+")
file_pos = f_pos.readlines()

f_neg = open("Dataset/neg.txt","r+")
file_neg = f_neg.readlines()

file = file_pos + file_neg

List = []

for x in file:
	x = x.lower()
	x = x.replace(".", " . ")
	x = x.replace(",", " , ")
	x = x.replace("-", " - ")
	x = x.replace("_", " _ ")
	x = x.replace("'", " ' ")
	x = x.translate({ord(i): None for i in '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'}).split(" ")
	y = []
	for i in x:
		if i:
			y.append(i);
	List.append(y)

random.shuffle(List)


w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

t = time()

w2v_model.build_vocab(List, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()

w2v_model.train(List, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.init_sims(replace=True)

print(w2v_model.wv.most_similar(positive=["good"], topn = 20))

print(w2v_model.wv.most_similar(positive=["bad"], topn = 20))