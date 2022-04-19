# import modules
from __future__ import annotations
from sre_constants import CATEGORY_LOC_WORD

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# load dataset
ds_spam = pd.read_csv("./Data/emails.csv")

# EDA and modify dataset, extract X_train, X_test, y_train, y_test
print(ds_spam.head(5).iloc[:,0])
print(ds_spam.tail(5).iloc[:,0])
print(ds_spam.describe())
print(ds_spam.isnull().sum())

# split data
X = ds_spam.iloc[:,0]
y = ds_spam.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.25)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# counter vectorization create vector matrix
#vectorizer = CountVectorizer()
#sample_text = ["He is a fool", "fools are not bad", "it is difficult to survive as a fool"]
#vector = vectorizer.fit_transform(sample_text)
#print("Vocabulary : ", vectorizer.vocabulary_)
#print(vector.toarray())
    # extract the words ignoring some words like "a" and arrange in alphabetical order 
    # like are, bad, fool, fools, he, is, not

vectorizer = CountVectorizer()
vector_X_train = vectorizer.fit_transform(X_train).toarray()

# model training
mnb = MultinomialNB()
mnb.fit(vector_X_train, y_train)
print(mnb)

# cross validation
#import seaborn as sns
#import matplotlib.pyplot as plt
y_train_predict = mnb.predict(vector_X_train)
print(y_train_predict)
cm = confusion_matrix(y_train, y_train_predict)
tn, fp, fn, tp = cm.ravel()
print(tn, fp, fn, tp)
P = tp / (tp+fp)
R = tp / (tp+fn)
F = 2 * P * R / (P+R)
print("F Score ", F)

#print(sns.heatmap(cm, annot=True))
#plt.plot(sns.heatmap(cm, annot=True))

# predict
test_sample = ["Free Money", "ANi, how are you"]
test_sample_vector = vectorizer.transform(test_sample).toarray()
vector_X_test = vectorizer.transform(X_test).toarray()

#print(test_sample_vector)
#print(vector_X_test)

#print(vector_X_train.shape)
#print(test_sample_vector.shape)
#print(vector_X_test.shape)

#test_predict = mnb.predict(test_sample_vector)
#y_test_predict = mnb.predict(vector_X_test)
#print(test_predict)


#print(X_test)
y_test_predict = mnb.predict(vector_X_test)
cm = confusion_matrix(y_test, y_test_predict)
tn, fp, fn, tp = cm.ravel()
print(tn, fp, fn, tp)
P = tp / (tp+fp)
R = tp / (tp+fn)
F = 2 * P * R / (P+R)
print("F Score Test ", F)
