
# coding: utf-8

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
import collections
import nltk
import sklearn
import re, string
#from sets import Set
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from bs4 import BeautifulSoup
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_classification
from nltk.stem.porter import PorterStemmer
from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_union
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
import pickle


# In[38]:


def getlabels(train):
    labels=[]
    temp=[]
    for i in range(0, train.size):
        temp=[train['toxic'][i],train['severe_toxic'][i],train['obscene'][i],train['threat'][i],train['insult'][i],train['identity_hate'][i]]
        labels.append(temp)
    return labels


# In[51]:


class Preprocessing:
    def __init__(self,comments,size):
        self.comments = comments
        self.cleaned_comments = []
        self.size = size
        self.vectorizer = None
    def clean_comments(self):
        for i in range(0, self.size):
            review_text = BeautifulSoup(self.comments[i]).get_text()  
            words = review_text.lower()  
            words=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",words) #removing user names
            words=re.sub("\[\[.*\]","",words)
            words = words.split()     
            snowball_stemmer = SnowballStemmer("english")
            meaningful_words = [snowball_stemmer.stem(word) for word in words]
            meaningful_words = " ".join(meaningful_words)
            self.cleaned_comments.append(meaningful_words)
    def vectorize(self):
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',
            ngram_range=(1, 1),
            max_features=28000)
        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='char',
            ngram_range=(1, 4),
            max_features=28000)
        self.vectorizer = make_union(word_vectorizer, char_vectorizer, n_jobs=2)
        return self.vectorizer
    def fit(self, data):
        return self.vectorizer.fit(data)
    def transform(self, data):
        return self.vectorizer.transform(data)
        
        
        
    
        


# In[85]:


class MyLogisticRegression:
    def __init__(self,categories,max_iter):
        self.params = {
                  'C'             : [1, 0.2, 0.6, 0.2, 0.45, 0.25],
                  'fit_intercept' : [True, True, True, True, True, True],
                  'penalty'       : ['l2', 'l2', 'l2', 'l2', 'l2', 'l2'],
                  'class_weight'  : [None, 'balanced', 'balanced', 'balanced', 'balanced', 'balanced'],
                 }
        self.categories = categories
        self.max_iter = max_iter
        self.dual = False
        self.classif = None
        self.l = []
    def fit_and_predict(self, x_train ,y_train_t,test_vecs):
        for index, category in enumerate(self.categories):
            self.classif = OneVsRestClassifier(LogisticRegression(C=self.params['C'][index],
            max_iter = self.max_iter,
            fit_intercept=self.params['fit_intercept'][index],
            penalty=self.params['penalty'][index],
            dual = self.dual,
            class_weight=self.params['class_weight'][index],
            verbose=0))
            self.classif.fit(x_train, y_train_t[index])
            pickle_out = open(str(category) + "_" + "pickle","wb")
            pickle.dump(self.classif, pickle_out)
            pickle_out.close()
            self.l.append((self.classif.predict_proba(test_vecs)[:,1]))
        return self.l

        
        
        
        


# In[64]:


if __name__ == "__main__":
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    labels=[]
    temp=[]
    for i in range(0, train['comment_text'].size):
        temp=[train['toxic'][i],train['severe_toxic'][i],train['obscene'][i],train['threat'][i],train['insult'][i],train['identity_hate'][i]]
        labels.append(temp)
        
    labels = np.asarray(labels)
        
    train_preprocessor = Preprocessing(train['comment_text'],train['comment_text'].size)
    test_preprocessor = Preprocessing(test['comment_text'],test['comment_text'].size)
    train_preprocessor.clean_comments()
    test_preprocessor.clean_comments()
    train_vectorizer = train_preprocessor.vectorize()
    #test_vectorizer = test_preprocessor.vectorize()
    train_vectorizer = train_preprocessor.fit(train_preprocessor.cleaned_comments)
    vectorized_train_vecs = train_vectorizer.transform(train_preprocessor.cleaned_comments)
    vectorized_test_vecs = train_vectorizer.transform(test_preprocessor.cleaned_comments)
    x_train, x_test, y_train, y_test =   cross_validation.train_test_split(vectorized_train_vecs,  labels, test_size=0.00)
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult' ,'identity_hate']
    max_iter = 200
    model = MyLogisticRegression(categories,max_iter)
    l = model.fit_and_predict(x_train,y_train.T,vectorized_test_vecs)
    id_numbers = list(test['Id'])
    temp_2 = ['Id'] + categories
    print(temp_2)

    f = open('sub_file.csv', 'w')
    w = csv.writer(f, delimiter=',')
    w.writerow(temp_2)
    
    for i in range(0, len(id_numbers)):
         #temp_l = [id_numbers[i], jff(l[0][i]), jff(l[1][i]), jff(l[2][i]), jff(l[3][i]), jff(l[4][i]), jff(l[5][i])]
        temp_l = [id_numbers[i], (l[0][i]), (l[1][i]), (l[2][i]), (l[3][i]), (l[4][i]), (l[5][i])]
         #print(temp_l)
        w.writerow(temp_l)

    print("completed")
    

