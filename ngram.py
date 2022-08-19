import csv
import re
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import pickle
from scipy.special import softmax
import numpy as np

# The text is obtained by concatenating the abstract to the title and removing wierd characters with the following function:
# def clean(strs):
#   strs = re.sub(r'[^a-zA-Z. ]', ' ', strs)
#   strs = re.sub(' +', ' ', strs)
#   return strs

train = pd.read_csv('Data/Processed/train.csv')
test = pd.read_csv('Data/Processed/test.csv')
trainX = train['text'].to_numpy()
trainY = train['label'].to_numpy()
testX = test['text'].to_numpy()
testY = test['label'].to_numpy()
def ngram_classifier (docs_train, y_train, docs_test, ngram_range, ):

    tfidfvec = TfidfVectorizer(stop_words = "english",
                                analyzer = 'word',
                                lowercase = True,
                                use_idf = True,
                                ngram_range = ngram_range)
    
    X_train = tfidfvec.fit_transform(docs_train)
    with open('encoder.pickle', 'wb') as handle:
        pickle.dump(tfidfvec, handle, protocol=pickle.HIGHEST_PROTOCOL)
    X_test = tfidfvec.transform(docs_test)
    
    
    clf = SGDClassifier(loss = "hinge", penalty = "l1")
    
    clf.fit(X_train, y_train)

    with open('model.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    prediction = softmax(clf.decision_function(X_test), axis=1)
    # prediction = clf.predict(X_test)
    return prediction

predY = ngram_classifier(trainX, trainY, testX, (1,1))
predY = np.argmax(predY, axis=1)

# print(predY)
print(accuracy_score(predY, testY))
# # Ngram accuracy is 78% 

print(confusion_matrix(predY, testY, labels=None, sample_weight=None, normalize=None))

