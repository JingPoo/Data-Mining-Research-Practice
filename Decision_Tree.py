#!/usr/bin/env python
# coding: utf-8

# In[1]:


#可以嘗試改善資料前處理部分&調整決策樹參數以增加Accuracy
#1:read data

import pandas as pd
df = pd.read_csv('character-deaths.csv')
df


# In[2]:


#2-1:處理空值

df = df.fillna(0)
df


# In[3]:


#2-2:取Death Year，將有數值轉成1

for i in range(len(df['Death Year'])):
    if df['Death Year'][i] > 0:
        df['Death Year'][i] = 1
#刪掉其餘兩個死亡相關的屬性
df = df.drop(columns=['Book of Death','Death Chapter'])
df


# In[4]:


#2-3:將Allegiances轉乘dummy特徵，方便後續預測

pd.set_option('display.max_columns', None) #show all columns
df = pd.get_dummies(df,columns=['Allegiances'])


# In[5]:


#2-4:亂數拆成訓練集(75%)與測試集(25%)

from sklearn.model_selection import train_test_split
X = df[['Book Intro Chapter','Gender','Nobility','GoT','CoK','SoS','FfC','DwD',
        'Allegiances_Arryn','Allegiances_Baratheon','Allegiances_Greyjoy','Allegiances_House Arryn',
        'Allegiances_House Baratheon','Allegiances_House Greyjoy','Allegiances_House Lannister',
        'Allegiances_House Martell','Allegiances_House Stark','Allegiances_House Targaryen',
        'Allegiances_House Tully','Allegiances_House Tyrell','Allegiances_Lannister',
        'Allegiances_Martell','Allegiances_Night\'s Watch','Allegiances_None',
        'Allegiances_Stark','Allegiances_Targaryen','Allegiances_Tully',
        'Allegiances_Tyrell','Allegiances_Wildling']]
y = df['Death Year']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
X_train


# In[6]:


#3:用DecisionTreeClassifier進行預測

from sklearn.tree import DecisionTreeClassifier
#經測試，樹深限制13準確度最高
clf = DecisionTreeClassifier(max_depth=13,random_state=42)
GoT_clf = clf.fit(X_train, y_train)
y_pred = GoT_clf.predict(X_test)


# In[7]:


#4:產生混淆矩陣，並計算Precision,Recall,Accuracy

from sklearn import metrics
print('Confusion Matrix:\n',metrics.confusion_matrix(y_test,y_pred))
print('Precision:',metrics.precision_score(y_test,y_pred))
print('Recall:',metrics.recall_score(y_test,y_pred))
print('Accuracy:',metrics.accuracy_score(y_test,y_pred))


# In[8]:


#5:產生決策樹的圖

from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(GoT_clf, out_file=None, max_depth=5) 
graph = graphviz.Source(dot_data)
graph.render("GoT_tree") 

