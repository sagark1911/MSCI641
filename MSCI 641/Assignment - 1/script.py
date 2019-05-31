import sys
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
import numpy as np

nltk.download("stopwords")

input_path = sys.argv[1]


f = open(input_path,"r+")
file = f.readlines()

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

train_list, rest = train_test_split(List, test_size = 0.2, random_state = 42)
val_list, test_list = train_test_split(rest, test_size = 0.5, random_state = 42)


List_noStopwords = []
stop_words = stopwords.words('english')

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
	y = [word for word in y if word not in stop_words]
	List_noStopwords.append(y)

train_list_no_stopword, rest_no_stopword = train_test_split(List_noStopwords, test_size = 0.2, random_state = 42)
val_list_no_stopword, test_list_no_stopword = train_test_split(rest_no_stopword, test_size = 0.5, random_state = 42)




np.savetxt("train.csv", train_list, delimiter=",", fmt='%s')
np.savetxt("val.csv", val_list, delimiter=",", fmt='%s')
np.savetxt("test.csv", test_list, delimiter=",", fmt='%s')

np.savetxt("train_no_stopword.csv", train_list_no_stopword, delimiter=",", fmt='%s')
np.savetxt("val_no_stopword.csv", val_list_no_stopword, delimiter=",", fmt='%s')
np.savetxt("test_no_stopword.csv", test_list_no_stopword, delimiter=",", fmt='%s')