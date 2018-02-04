import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
import os
import csv,sys
import glob, os
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
import string
import nltk
from sklearn.naive_bayes import BernoulliNB


con = sqlite3.connect('database.sqlite')

messages = pd.read_sql_query("""
SELECT 
  Score, 
  Summary
FROM Reviews 
WHERE Score != 3""", con)

messages["Sentiment"] = messages["Score"].apply(lambda score: "positive" if score > 3 else "negative")


#print(pd.read_sql_query("SELECT * FROM Reviews LIMIT 3", con))


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
#print("%d items in training data, %d in test data" % (len(train), len(test)))
count_vect = CountVectorizer(min_df = 1, ngram_range = (1, 4))

X_train_counts = count_vect.fit_transform(train["Summary_Clean"])
X_new_counts = count_vect.transform(test["Summary_Clean"])


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

y_train = train["Sentiment"]
y_test = test["Sentiment"]
#print(X_train_tfidf)
prediction = dict()


logreg = LogisticRegression(C=1e5)
logreg_result = logreg.fit(X_train_tfidf, y_train)
prediction['Logistic'] = logreg.predict(X_test_tfidf)


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train_tfidf, y_train)
prediction['random'] = clf.predict(X_test_tfidf)


model = MultinomialNB().fit(X_train_tfidf, y_train)
prediction['Multinomial'] = model.predict(X_test_tfidf)



model = BernoulliNB().fit(X_train_tfidf, y_train)
prediction['Bernoulli'] = model.predict(X_test_tfidf)
'''
#Too much time
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_tfidf, y_train)
prediction['decision'] = clf.predict(X_test_tfidf)


#svm
#Takes a lot of time!!!

from sklearn import svm
clf=svm.SVC(kernel="linear")
model=clf.fit(X_train_tfidf,y_train)
prediction['SVM']=clf.predict(X_test_tfidf)
'''


def formatt(x):
    if x == 'negative':
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
colors = ['k', 'm', 'y', 'b', 'g']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.3])
plt.ylim([-0.1,1.3])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

pos=[]
neg=[]
def test_sample(model, sample):
	sample_counts = count_vect.transform([sample])
	sample_tfidf = tfidf_transformer.transform(sample_counts)
	result = model.predict(sample_tfidf)[0]
	prob = model.predict_proba(sample_tfidf)[0]
	#print("Sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))
	if prob[0] > prob[1]:
		neg.append(sample)
	else :
		pos.append(sample)
    

#What should the input be??? Again the dataset? What data are we segregating??
#Now its just 3 setences.
test_sample(logreg, "The food was delicious, it smelled great and the taste was awesome")
test_sample(logreg, "The whole experience was horrible. The smell was so bad that it literally made me sick.")
test_sample(logreg, "The food was ok, I guess. The smell wasn't very good, but the taste was ok.")

print("pos: ",pos)
print()
print("neg: ",neg)
