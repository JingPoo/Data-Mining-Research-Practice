
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv',sep='\t')
X_train = train_df.drop(columns='label')
X_train


# In[2]:


Y_train = train_df.drop(columns='text')
#label中有一異常值('label')，他非0,1
#將異常值改成0
Y_train.replace('label',0,inplace=True)
Y_train = Y_train['label']
Y_train


# In[3]:


X_test = pd.read_csv('test.csv',sep='\t')
X_test = X_test.drop(columns='id')
X_test


# In[4]:


test_label = pd.read_csv('sample_submission.csv')
Y_test = test_label.drop(columns='id')
Y_test = Y_test['label']
Y_test


# In[5]:


from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

#建立token
token = Tokenizer(num_words=3800)
token.fit_on_texts(X_train['text'])

#查看英文轉數字的結果
print(token.word_index)

#將[文字]轉換成[數字list]
x_train_seq = token.texts_to_sequences(X_train['text'])
x_test_seq = token.texts_to_sequences(X_test['text'])


# In[6]:


#截長補短
X_train = sequence.pad_sequences(x_train_seq,maxlen=380)
X_test = sequence.pad_sequences(x_test_seq,maxlen=380)
print(X_train)
print(X_test)


# In[7]:


from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN

#RNN 

#建立線性堆疊模型
modelRNN = Sequential()
#將[數字list]轉換成[向量list]
modelRNN.add(Embedding(output_dim=32,
                       input_dim=3800,
                       input_length=380))
#Dropout避免overfitting
modelRNN.add(Dropout(0.7))
#RNN層
modelRNN.add(SimpleRNN(units=16))
#隱藏層
modelRNN.add(Dense(units=256,activation='relu'))
#Dropout
modelRNN.add(Dropout(0.7))
#輸出層
modelRNN.add(Dense(units=1,activation='sigmoid'))


# In[8]:


modelRNN.summary()


# In[9]:


#訓練模型

#對訓練模型進行設定
modelRNN.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])


# In[10]:


train_history_RNN = modelRNN.fit(X_train, Y_train,
                                 epochs=10,
                                 batch_size=100,
                                 verbose=2,
                                 validation_split=0.2)


# In[11]:


#LSTM
from keras.layers.recurrent import LSTM

modelLSTM = Sequential()
modelLSTM.add(Embedding(output_dim=32,
                       input_dim=3800,
                       input_length=380))
modelLSTM.add(Dropout(0.7))
modelLSTM.add(LSTM(32))
modelLSTM.add(Dense(units=256,activation='relu'))
modelLSTM.add(Dropout(0.7))
modelLSTM.add(Dense(units=1,activation='sigmoid'))


# In[12]:


modelLSTM.summary()


# In[13]:


#對訓練模型進行設定
modelLSTM.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])


# In[14]:


train_history_LSTM = modelLSTM.fit(X_train, Y_train,
                                   epochs=10,
                                   batch_size=100,
                                   verbose=2,
                                   validation_split=0.2)


# In[15]:


# RNN
# plot出訓練過程中的Accuracy與Loss值變化
print('-----RNN-----')
plt.plot(train_history_RNN.history['acc'])
plt.plot(train_history_RNN.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(train_history_RNN.history['loss'])
plt.plot(train_history_RNN.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[16]:


#用test資料對模型計算準確度
scores = modelRNN.evaluate(X_test,Y_test,verbose=1)
print('RNN accuracy:',scores[1])


# In[17]:


# LSTM
# plot出訓練過程中的Accuracy與Loss值變化
print('-----LSTM-----')
plt.plot(train_history_LSTM.history['acc'])
plt.plot(train_history_LSTM.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(train_history_LSTM.history['loss'])
plt.plot(train_history_LSTM.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[18]:


scores = modelLSTM.evaluate(X_test,Y_test,verbose=1)
print('LSTM accuracy:',scores[1])

