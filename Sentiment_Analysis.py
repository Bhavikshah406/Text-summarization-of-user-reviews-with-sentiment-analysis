import sqlite3
import pandas as pd
import numpy as np
import nltk,re
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc ,roc_auc_score
from nltk.stem.porter import PorterStemmer
import csv,sys
import glob, os
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import string
import nltk
from sklearn.naive_bayes import BernoulliNB
from time import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm



con = sqlite3.connect('database.sqlite')

messages = pd.read_sql_query("""
SELECT 
  Score, 
  Summary
FROM Reviews 
WHERE Score != 3""", con)

messages["Sentiment"] = messages["Score"].apply(lambda score: "positive" if score > 3 else "negative")


cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    
    return sentence

'''
#with stopword removal takes more time and less accurate!!
def cleanup(sentence):
	sentence = sentence.lower()
	sentence = cleanup_re.sub(' ', sentence).strip() #Find all substrings where the RE matches, and replace them with a different string
	stop_words = set(stopwords.words("english"))
	word_tokens = word_tokenize(sentence)
	filtered_sentence = [w for w in word_tokens if not w in stop_words]
	#print(filtered_sentence)
	sentence = ''.join(filtered_sentence)
	print(sentence[0],sentence[1],sentence[2])
	return sentence
'''
messages["Summary_Clean"] = messages["Summary"].apply(cleanup)

train, test = train_test_split(messages, test_size=0.2)
count_vect = CountVectorizer(min_df = 1, ngram_range = (1, 4))

X_train_counts = count_vect.fit_transform(train["Summary_Clean"])
X_new_counts = count_vect.transform(test["Summary_Clean"])


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

y_train = train["Sentiment"]
y_test = test["Sentiment"]
RAN_STATE = 42
prediction_for_test = dict()
prediction_for_train = dict()
prediction_for_train_10 = dict()
prediction_for_train_20 = dict()

count1=1

def train_classifier(clf, X_train, y_train):
	global count1
	start=time()
	clf.fit(X_train, y_train)
	end=time()
	print("Trained model in {:.4f} seconds".format(end - start))
	
	if count1==1:
		prediction_for_train_10[clf.__class__.__name__] = clf.predict(X_test_tfidf)
		count1=count1+1
	if count1==2:
		prediction_for_train_20[clf.__class__.__name__] = clf.predict(X_test_tfidf)
		count1=count1+1
	else:
		prediction_for_test[clf.__class__.__name__] = clf.predict(X_test_tfidf)
		prediction_for_train[clf.__class__.__name__] = clf.predict(X_train_tfidf)
		count1=1
	
    


def train_wrap(clf, X_train, y_train, X_test, y_test):
	print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, X_train.shape[0]))
	train_classifier(clf, X_train, y_train)
   

    


clf1=MultinomialNB()
clf2=AdaBoostClassifier(random_state = RAN_STATE)
clf3=RandomForestClassifier(random_state = RAN_STATE)
clf4=LogisticRegression(C=1e5)
clf5= DecisionTreeClassifier(random_state=20160121, criterion='entropy')
clf6=clf=svm.SVC(kernel="linear")
clf_list = [clf1,clf2,clf4]



train_feature_list = [X_train_tfidf[0:100000],X_train_tfidf[0:200000],X_train_tfidf]
train_target_list = [y_train[0:100000], y_train[0:200000], y_train]



for clf in clf_list:
    for a, b in zip(train_feature_list, train_target_list):
        train_wrap(clf, a, b, X_test_tfidf, y_test)


def formatt(x):
    if x == 'negative':
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
best_model=0
best_auc=0
colors = ['k', 'm', 'y', 'b', 'g', 'k']
cnt=0
def roc(whatx,whaty,what):
	global cmp
	global  cnt
	for model, predicted in whatx.items():
		false_positive_rate, true_positive_rate, thresholds = roc_curve(whaty.map(formatt), vfunc(predicted))
		roc_auc = auc(false_positive_rate, true_positive_rate)
		if roc_auc > best_auc:
			best_model=cnt
		#print(what, roc_auc)
		plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
		cmp += 1
		cnt += 1
	cnt=0
	cmp=0

	plt.title('Classifiers comparaison with ROC for '+what)
	plt.legend(loc='lower right')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([-0.1,1.3])
	plt.ylim([-0.1,1.3])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()


roc(prediction_for_train_10,y_test,"test prediction training of for 100000")
roc(prediction_for_train_20,y_test,"test prediction training of for 200000")	
roc(prediction_for_train,y_train,"train dataset")
roc(prediction_for_test,y_test,"test dataset")



pos=[]
neg=[]
def test_sample(model, sample):
	sample_counts = count_vect.transform([sample])
	sample_tfidf = tfidf_transformer.transform(sample_counts)
	result = model.predict(sample_tfidf)[0]
	prob = model.predict_proba(sample_tfidf)[0]
	print("Sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))
	if prob[0] > prob[1]:
		neg.append(sample)
	else :
		pos.append(sample)
    


test_sample(clf_list[best_model], "The food was very tasty")
test_sample(clf_list[best_model], "The whole experience was horrible. The smell was so bad that it literally made me sick.")
test_sample(clf_list[best_model], "The food was bad.")

print("positive reviews are: ",pos)
print()
print("negative reviews are: ",neg)
