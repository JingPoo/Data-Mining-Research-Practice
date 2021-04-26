#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train_df = pd.read_csv('train.csv',sep='\t')
train = train_df.drop(columns='label')
train


# In[2]:


Y_train = train_df.drop(columns='text')
Y_train


# In[3]:


test = pd.read_csv('test.csv',sep='\t')
test


# In[4]:


test_label = pd.read_csv('sample_submission.csv')
Y_test = test_label.drop(columns='id')
Y_test


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#train
vectorizer = CountVectorizer(stop_words='english',max_features=32827)
X = vectorizer.fit_transform(train['text'])

print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names())
print(X.toarray())


# In[6]:


transformer = TfidfTransformer()
X_train = transformer.fit_transform(X)
print(X_train.toarray())
print(vectorizer.get_feature_names())


# In[7]:


#test
vectorizer = CountVectorizer(stop_words='english',max_features=32827)
X = vectorizer.fit_transform(test['text'])

print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names())
print(X.toarray())


# In[8]:


transformer = TfidfTransformer()
X_test = transformer.fit_transform(X)
print(X_test.toarray())
print(vectorizer.get_feature_names())


# In[9]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#xgboost
from xgboost import XGBClassifier

XGBmodel = XGBClassifier()
XGBmodel.fit(X_train,Y_train)
print(XGBmodel)


# In[10]:


Y_pred = XGBmodel.predict(X_test)
Y_pred = Y_pred.astype('int')
Y_pred


# In[11]:


print('XGB')
acc = accuracy_score(Y_test,Y_pred)
print('Accuracy:',acc)
pre = precision_score(Y_test,Y_pred)
print('Precision:',pre)
rec = recall_score(Y_test,Y_pred)
print('Recall:',rec)
f = f1_score(Y_test,Y_pred)
print('F-measure:',f)


# In[12]:


#GBDT
from sklearn.ensemble import GradientBoostingClassifier
GBDTmodel = GradientBoostingClassifier()
GBDTmodel.fit(X_train,Y_train)
print(GBDTmodel)


# In[13]:


Y_pred = GBDTmodel.predict(X_test)
Y_pred = Y_pred.astype('int')
Y_pred


# In[14]:


print('GBDT')
acc = accuracy_score(Y_test,Y_pred)
print('Accuracy:',acc)
pre = precision_score(Y_test,Y_pred)
print('Precision:',pre)
rec = recall_score(Y_test,Y_pred)
print('Recall:',rec)
f = f1_score(Y_test,Y_pred)
print('F-measure:',f)


# In[15]:


from lightgbm import LGBMClassifier


# In[16]:


LGBMmodel = LGBMClassifier()
LGBMmodel.fit(X_train,Y_train)
print(LGBMmodel)


# In[17]:


Y_pred = LGBMmodel.predict(X_test)
Y_pred = Y_pred.astype('int')
Y_pred


# In[18]:


print('LGBM')
acc = accuracy_score(Y_test,Y_pred)
print('Accuracy:',acc)
pre = precision_score(Y_test,Y_pred)
print('Precision:',pre)
rec = recall_score(Y_test,Y_pred)
print('Recall:',rec)
f = f1_score(Y_test,Y_pred)
print('F-measure:',f)


# In[ ]:




