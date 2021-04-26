#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('HW2data.csv')
df


# In[2]:


#資料前處理

# deal with ?
df['workclass'] = df['workclass'].str.replace('?',str(Counter(df['workclass']).most_common(1)[0][0]))
df['occupation'] = df['occupation'].str.replace('?',str(Counter(df['occupation']).most_common(1)[0][0]))
df['native_country'] = df['native_country'].str.replace('?',str(Counter(df['native_country']).most_common(1)[0][0]))
#0,1
df['sex'] = df['sex'].map({' Male':0,' Female':1}).astype(int)
df['income'] = df['income'].map({' <=50K':0,' >50K':1}).astype(int)
# deal with string value
df = pd.get_dummies(df)
df


# In[3]:


X = pd.concat([df.iloc[:,:7],df.iloc[:,8:]],axis=1)
y = df['income']


# In[4]:


#用Random Forest分類
rf = RandomForestClassifier(n_estimators=100,max_depth=20,random_state=1)
#K-fold cross-validation
def K_fold_CV(k,data):
    #設定subset size 即data長度/k
    #設定Accuracy初始值
    subset_size = int(len(data)/k)
    Accuracy = 0
    for i in range(k):
        #設定testing set與training set的資料起始點與結束點
        #例如資料有100筆，testing set在本次iteration取第1到10筆，則training set為第11到100筆；下次testing set為11~20，training set為21~100 & 1~10
        #利用training set建立模型，testing set計算出Accuracy累加
        X_test = X.iloc[i*subset_size+1:(i+1)*subset_size,:]
        y_test = y.iloc[i*subset_size+1:(i+1)*subset_size]
        X_train = X.drop(df.index[int(i*subset_size+1):int((i+1)*subset_size)])
        y_train = y.drop(df.index[int(i*subset_size+1):int((i+1)*subset_size)])
        model = rf.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        Accuracy += accuracy_score(y_test,y_pred)
    return Accuracy/k

K_fold_CV(10,df)

