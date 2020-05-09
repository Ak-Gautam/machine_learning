# coding: utf-8

# ML project using the weather data of bangalore
#Made to be executed on google colab

# In[ ]:


import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras
import pandas as pd


# In[2]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[ ]:
#Loading the dataset from the scv file

#df = pd.read_csv('/content/gdrive/My Drive/bengaluru.csv')


# In[ ]:


#df.head()


# In[ ]:


#df.keys()


# In[ ]:


columns_to_keep = ['date_time',
                   'maxtempC',
                   'mintempC',
                   'sunHour',
                   'cloudcover',
                   'humidity',
                   'pressure',
                   'tempC']


# In[ ]:


df = df[columns_to_keep]
#df.head()


# In[ ]:


df = df.set_index("date_time")
#df.head(1)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import MinMaxScaler
import warnings


# In[ ]:


train, test = df[:-25000], df[-25000:]


# In[ ]:


scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)


# In[12]:


n_features = 7
n_input = 12

generator = keras.preprocessing.sequence.TimeseriesGenerator(train, train, length = n_input, batch_size = 300)

model = tf.keras.Sequential([keras.layers.Dense(units=7)])
model.add(keras.layers.LSTM(7, activation='relu', input_shape = (n_input, n_features)))
model.add(keras.layers.Dropout(0.05))
model.compile(optimizer = 'adam', loss = 'mse')

model.fit(generator, epochs = 30)


# In[ ]:


pred_list1 = []
batch1 = train[-n_input:].reshape((1, n_input, n_features))

for i in range(30):
    pred_list1.append(model.predict(batch1)[0])
    batch1 = np.append(batch1[:, 1:,:], [[pred_list1[i]]], axis = 1)


# In[14]:


np.shape(scaler.inverse_transform(pred_list1))


# In[ ]:


dfpredict = pd.DataFrame(scaler.inverse_transform(pred_list1), index = df[-30:].index, columns=['PtempM','Ptempm', 'PsunH','PcloudH','Phum','Ppressure','Ptemp'])


# In[ ]:


df_test = pd.concat([df, dfpredict], axis = 1)


# In[ ]:


#df_test.tail(31)


# In[ ]:


dtest = df_test[-30:]
#dtest.head()


# In[19]:


plt.figure(figsize=(20,5))
plt.plot(dtest.index, dtest['tempC'])
plt.plot(dtest.index, dtest['Ptemp'], color = 'r')
plt.show


# In[20]:


add_dates = [pd.Timestamp(df.index[-1]) + pd.tseries.offsets.DateOffset(days = x) for x in range(0,151)]
future_dates = pd.DataFrame(index = add_dates[1:], columns = df.columns)
future_dates.tail()


# In[ ]:


df_predict = pd.DataFrame(scaler.inverse_transform(pred_list1),
                index = future_dates[-30:].index, columns = ['PtempM','Ptempm','PsunH','PcloudH','Phum','Ppressure','Ptemp'])

df_proj = pd.concat([df, df_predict], axis = 1)


# In[23]:


df_proj.tail(40)


# In[ ]:




