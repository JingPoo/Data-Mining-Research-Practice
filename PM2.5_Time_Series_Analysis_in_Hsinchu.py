#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv('新竹_2019.csv',encoding='Big5')
data.columns = data.columns.str.strip() #去除columns內的多餘空白
data


# In[2]:


#1.資料前處理
#a. 取出10.11.12月資料

data = data.iloc[4825:,:]
data


# In[3]:


#b. 缺失值以及無效值以前後一小時平均值取代

#找出前面有效值的col位置
def recursiveCheckLeft(i,j,v1,v2,v3,v4):
    #判斷當col=3時，找前18row的最後一個值
    if j != 3:
        if data.iloc[i,j-1].find(v1)==-1 and data.iloc[i,j-1].find(v2)==-1 and data.iloc[i,j-1].find(v3)==-1 and data.iloc[i,j-1].find(v4)==-1: #前個資料是有效的
            #print('left value:',data.iloc[i,j-1])
            return j-1
        else: #前個資料是無效值
            return recursiveCheckLeft(i,j-1,v1,v2,v3,v4)
    else: #到0時了
        return recursiveCheckLeft(i-18,j+23,v1,v2,v3,v4) #看前一天23時的值
    
#找出後面有效值的col位置
def recursiveCheckRight(i,j,v1,v2,v3,v4):
    #判斷當col=26時，找後18row的第一個值
    if j != 26:
        if data.iloc[i,j+1].find(v1)==-1 and data.iloc[i,j+1].find(v2)==-1 and data.iloc[i,j+1].find(v3)==-1 and data.iloc[i,j+1].find(v4)==-1: #後個資料是有效的
            #print('right value:',data.iloc[i,j+1])
            return j+1
        else: #後個資料是無效值
            return recursiveCheckRight(i,j+1,v1,v2,v3,v4)
    else: #到23時了
        return recursiveCheckLeft(i+18,j-23,v1,v2,v3,v4) #看後一天0時的值   


# In[4]:


for i in range(data.shape[0]): #row
    for j in range(3,data.shape[1]): #col  
        if data.iloc[i,j].find('#')!=-1: #若為無效值(不包含#會回傳-1)
            left = recursiveCheckLeft(i,j,'#','*','x','A')
            right = recursiveCheckRight(i,j,'#','*','x','A')
            #用前後有效值的平均去取代
            replace_value = ( float(data.iloc[i,left]) + float(data.iloc[i,right]) )/2
            data.iloc[i,j] = str(replace_value)
        elif data.iloc[i,j].find('*')!=-1:
            left = recursiveCheckLeft(i,j,'#','*','x','A')
            right = recursiveCheckRight(i,j,'#','*','x','A')
            replace_value = ( float(data.iloc[i,left]) + float(data.iloc[i,right]) )/2
            data.iloc[i,j] = str(replace_value)
        elif data.iloc[i,j].find('x')!=-1: 
            left = recursiveCheckLeft(i,j,'#','*','x','A')
            right = recursiveCheckRight(i,j,'#','*','x','A')
            replace_value = ( float(data.iloc[i,left]) + float(data.iloc[i,right]) )/2
            data.iloc[i,j] = str(replace_value)
        elif data.iloc[i,j].find('A')!=-1: 
            left = recursiveCheckLeft(i,j,'#','*','x','A')
            right = recursiveCheckRight(i,j,'#','*','x','A')
            replace_value = ( float(data.iloc[i,left]) + float(data.iloc[i,right]) )/2
            data.iloc[i,j] = str(replace_value)

data


# In[5]:


#d. 將資料切割成訓練集(10.11月)以及測試集(12月)

training_set = data.iloc[:1098,:]
testing_set = data.iloc[1098:,:]
training_set


# In[6]:


#e. 製作時序資料: 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料

#把不要的col拿掉
training_set = training_set.iloc[:,2:]
testing_set = testing_set.iloc[:,2:]

#製作時序資料
left = training_set.iloc[:18,:] #取出第一天的資料
for i in range(18,training_set.shape[0],18): #每18行合併
    right = training_set.iloc[i:(i+18),:]
    left = pd.merge(left, right, on='測項')
time_trainging_set = left

left = testing_set.iloc[:18,:] #取出第一天的資料
for i in range(18,testing_set.shape[0],18): #每18行合併
    right = testing_set.iloc[i:(i+18),:]
    left = pd.merge(left, right, on='測項')
time_testing_set = left

#PM2.5 row轉為int，其他轉為float ->用於Regression
time_trainging_set.iloc[:,1:] = time_trainging_set.iloc[:,1:].astype('float')
time_trainging_set.iloc[9,1:] = time_trainging_set.iloc[9,1:].astype('int')
time_testing_set.iloc[:,1:] = time_testing_set.iloc[:,1:].astype('float')
time_testing_set.iloc[9,1:] = time_testing_set.iloc[9,1:].astype('int')


# In[7]:


#2.時間序列
#預測目標

#將未來第一個小時當預測目標
X1_PM=[] # X_train for only pm2.5
X1_ALL=[] # X_train for all attributes
Y1=[] # y_train
X1_PM_test=[] # X_test for only pm2.5
X1_ALL_test=[] # X_test for all attributes
Y1_test=[] # y_test

#將未來第六個小時當預測目標
X2_PM=[] 
X2_ALL=[]
Y2=[]
X2_PM_test=[]
X2_ALL_test=[]
Y2_test=[]

# X1_PM,X1_ALL
for i in range(1,time_trainging_set.shape[1]-6):
    X1_PM_values=[] #暫放每一格的5個數值
    X1_ALL_values=[] #暫放每一格的18*6個數值
    for j in range(i,i+6):
        X1_PM_values.append(time_trainging_set.iloc[9,j])
        for k in range(time_trainging_set.shape[0]): 
            X1_ALL_values.append(time_trainging_set.iloc[k,j])
    X1_PM.append(X1_PM_values)
    X1_ALL.append(X1_ALL_values)
    
# X1_PM_test,X1_ALL_test
for i in range(1,time_testing_set.shape[1]-6):
    X1_PM_test_values=[] 
    X1_ALL_test_values=[] 
    for j in range(i,i+6):
        X1_PM_test_values.append(time_testing_set.iloc[9,j]) 
        for k in range(time_testing_set.shape[0]): 
            X1_ALL_test_values.append(time_testing_set.iloc[k,j])
    X1_PM_test.append(X1_PM_test_values)
    X1_ALL_test.append(X1_ALL_test_values)
    
# X2_PM,X2_ALL
for i in range(1,time_trainging_set.shape[1]-11):
    X2_PM_values=[] 
    X2_ALL_values=[] 
    for j in range(i,i+6):
        X2_PM_values.append(time_trainging_set.iloc[9,j])
        for k in range(time_trainging_set.shape[0]): 
            X2_ALL_values.append(time_trainging_set.iloc[k,j])
    X2_PM.append(X2_PM_values)
    X2_ALL.append(X2_ALL_values)
    
# X2_PM_test,X2_ALL_test
for i in range(1,time_testing_set.shape[1]-11):
    X2_PM_test_values=[] 
    X2_ALL_test_values=[] 
    for j in range(i,i+6):
        X2_PM_test_values.append(time_testing_set.iloc[9,j])
        for k in range(time_testing_set.shape[0]): 
            X2_ALL_test_values.append(time_testing_set.iloc[k,j])
    X2_PM_test.append(X2_PM_test_values)
    X2_ALL_test.append(X2_ALL_test_values)

# Y1,Y2
for i in range(7,time_trainging_set.shape[1]):
        #將未來第一個小時當預測目標
        Y1.append(time_trainging_set.iloc[9,i])
        #將未來第六個小時當預測目標
        if i < time_trainging_set.shape[1]-5:
            Y2.append(time_trainging_set.iloc[9,i+5]) 

# Y1_test,Y2_test
for i in range(7,time_testing_set.shape[1]):
        #將未來第一個小時當預測目標
        Y1_test.append(time_testing_set.iloc[9,i])
        #將未來第六個小時當預測目標
        if i < time_testing_set.shape[1]-5:
            Y2_test.append(time_testing_set.iloc[9,i+5]) 


# In[8]:


#change list to np.array

import numpy as np

X1_PM = np.array(X1_PM)
X1_ALL = np.array(X1_ALL)
Y1 = np.array(Y1)
X1_PM_test = np.array(X1_PM_test)
X1_ALL_test = np.array(X1_ALL_test)
Y1_test = np.array(Y1_test)

X2_PM = np.array(X2_PM)
X2_ALL = np.array(X2_ALL)
Y2 = np.array(Y2)
X2_PM_test = np.array(X2_PM_test)
X2_ALL_test = np.array(X2_ALL_test)
Y2_test = np.array(Y2_test)


# In[9]:


#c. 使用兩種模型 Linear Regression 和 Random Forest Regression 建模

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#linear Regression
Lmodel_1P = LinearRegression().fit(X1_PM,Y1)
Lmodel_1A = LinearRegression().fit(X1_ALL,Y1)
Lmodel_2P = LinearRegression().fit(X2_PM,Y2)
Lmodel_2A = LinearRegression().fit(X2_ALL,Y2)

#Random Forest Regression
rf1 = RandomForestRegressor(n_estimators=100,max_depth=20,random_state=1)
Rmodel_1P = rf1.fit(X1_PM,Y1)
rf2 = RandomForestRegressor(n_estimators=100,max_depth=20,random_state=1)
Rmodel_1A = rf2.fit(X1_ALL,Y1)
rf3 = RandomForestRegressor(n_estimators=100,max_depth=20,random_state=1)
Rmodel_2P = rf3.fit(X2_PM,Y2)
rf4 = RandomForestRegressor(n_estimators=100,max_depth=20,random_state=1)
Rmodel_2A = rf4.fit(X2_ALL,Y2)


# In[10]:


#d. 用測試集資料計算MAE (會有8個結果， 2種X資料 * 2種Y資料 * 2種模型)

from sklearn.metrics import mean_absolute_error

#linear Regression
    #PM2.5,predict 6
y_pred_Lmodel_1P = Lmodel_1P.predict(X1_PM_test)
mae1 = mean_absolute_error(Y1_test,y_pred_Lmodel_1P)
    #ALL,predict 6
y_pred_Lmodel_1A = Lmodel_1A.predict(X1_ALL_test)
mae2 = mean_absolute_error(Y1_test,y_pred_Lmodel_1A)
    #PM2.5,predict 11
y_pred_Lmodel_2P = Lmodel_2P.predict(X2_PM_test)
mae3 = mean_absolute_error(Y2_test,y_pred_Lmodel_2P)
    #ALL,predict 11
y_pred_Lmodel_2A = Lmodel_2A.predict(X2_ALL_test)
mae4 = mean_absolute_error(Y2_test,y_pred_Lmodel_2A)

#Random Forest Regression
    #PM2.5,predict 6
y_pred_Rmodel_1P = Rmodel_1P.predict(X1_PM_test)
mae5 = mean_absolute_error(Y1_test,y_pred_Rmodel_1P)
    #ALL,predict 6
y_pred_Rmodel_1A = Rmodel_1A.predict(X1_ALL_test)
mae6 = mean_absolute_error(Y1_test,y_pred_Rmodel_1A)
    #PM2.5,predict 11
y_pred_Rmodel_2P = Rmodel_2P.predict(X2_PM_test)
mae7 = mean_absolute_error(Y2_test,y_pred_Rmodel_2P)
    #ALL,predict 11
y_pred_Rmodel_2A = Rmodel_2A.predict(X2_ALL_test)
mae8 = mean_absolute_error(Y2_test,y_pred_Rmodel_2A)

print('',mae1,'\n',mae2,'\n',mae3,'\n',mae4,'\n',mae5,'\n',mae6,'\n',mae7,'\n',mae8)

