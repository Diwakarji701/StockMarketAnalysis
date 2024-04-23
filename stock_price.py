import yfinance as yf
from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

stock = "GOOG"
google_data = yf.download(stock, start, end)

google_data.head()


# In[6]:


google_data.shape


# In[7]:


google_data.describe()


# In[8]:


google_data.info()


# In[9]:


google_data.isna().sum()


# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


plt.figure(figsize = (15,5))
google_data['Adj Close'].plot()
plt.xlabel("years")
plt.ylabel("Adj Close")
plt.title("Closing price of Google data")


# In[13]:


def plot_graph(figsize, values, column_name):
    plt.figure()
    values.plot(figsize = figsize)
    plt.xlabel("years")
    plt.ylabel(column_name)
    plt.title(f"{column_name} of Google data")
    


# In[14]:


google_data.columns


# In[15]:


for column in google_data.columns:
    plot_graph((15,5),google_data[column], column)


# In[ ]:


# 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

# MA for 5 days ==> null null null null 30 40 50 60 70 80


# In[16]:


temp_data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(sum(temp_data[1:6])/5)


# In[17]:


import pandas as pd
data = pd.DataFrame([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
data.head()


# In[18]:


data['MA'] = data.rolling(5).mean()
data


# In[19]:


for i in range(2004,2025):
    print(i,list(google_data.index.year).count(i))


# In[20]:


google_data['MA_for_250_days'] = google_data['Adj Close'].rolling(250).mean()


# In[21]:


google_data['MA_for_250_days'][0:250].tail()


# In[22]:


plot_graph((15,5), google_data['MA_for_250_days'], 'MA_for_250_days')


# In[23]:


plot_graph((15,5), google_data[['Adj Close','MA_for_250_days']], 'MA_for_250_days')


# In[24]:


google_data['MA_for_100_days'] = google_data['Adj Close'].rolling(100).mean()
plot_graph((15,5), google_data[['Adj Close','MA_for_100_days']], 'MA_for_100_days')


# In[25]:


plot_graph((15,5), google_data[['Adj Close','MA_for_100_days', 'MA_for_250_days']], 'MA')


# In[26]:


google_data['percentage_change_cp'] = google_data['Adj Close'].pct_change()
google_data[['Adj Close','percentage_change_cp']].head()


# In[27]:


plot_graph((15,5), google_data['percentage_change_cp'], 'percentage_change')


# In[28]:


Adj_close_price = google_data[['Adj Close']]


# In[29]:


max(Adj_close_price.values),min(Adj_close_price.values) 


# In[30]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(Adj_close_price)
scaled_data


# In[31]:


len(scaled_data)


# In[32]:


x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
    
import numpy as np
x_data, y_data = np.array(x_data), np.array(y_data)


# In[33]:


x_data[0],y_data[0]


# In[34]:


int(len(x_data)*0.7)


# In[35]:


4908-100-int(len(x_data)*0.7)


# In[36]:


splitting_len = int(len(x_data)*0.7)
x_train = x_data[:splitting_len]
y_train = y_data[:splitting_len]

x_test = x_data[splitting_len:]
y_test = y_data[splitting_len:]


# In[37]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[38]:


from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[39]:


model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[40]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[41]:


model.fit(x_train, y_train, batch_size=1, epochs = 2)


# In[42]:


model.summary()


# In[43]:


predictions = model.predict(x_test)


# In[44]:


predictions


# In[45]:


inv_predictions = scaler.inverse_transform(predictions)
inv_predictions


# In[46]:


inv_y_test = scaler.inverse_transform(y_test)
inv_y_test


# In[47]:


rmse = np.sqrt(np.mean( (inv_predictions - inv_y_test)**2))


# In[48]:


rmse


# In[49]:


ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_predictions.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
ploting_data.head()


# In[50]:


plot_graph((15,6), ploting_data, 'test data')


# In[57]:


plot_graph((30,10), pd.concat([Adj_close_price[:splitting_len+100],ploting_data], axis=0), 'whole data')


# In[ ]:





# In[52]:


model.save("Latest_stock_price_model.keras")


# In[ ]:




