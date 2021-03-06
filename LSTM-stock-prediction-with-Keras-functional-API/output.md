```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from yahoofinancials import YahooFinancials
%matplotlib inline
```


```python
appl_df = yf.download('AAPL', 
                      start='2018-01-01', 
                      end='2019-12-31', 
                      progress=False)
appl_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02</th>
      <td>42.540001</td>
      <td>43.075001</td>
      <td>42.314999</td>
      <td>43.064999</td>
      <td>41.310070</td>
      <td>102223600</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>43.132500</td>
      <td>43.637501</td>
      <td>42.990002</td>
      <td>43.057499</td>
      <td>41.302879</td>
      <td>118071600</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>43.134998</td>
      <td>43.367500</td>
      <td>43.020000</td>
      <td>43.257500</td>
      <td>41.494736</td>
      <td>89738400</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>43.360001</td>
      <td>43.842499</td>
      <td>43.262501</td>
      <td>43.750000</td>
      <td>41.967163</td>
      <td>94640000</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>43.587502</td>
      <td>43.902500</td>
      <td>43.482498</td>
      <td>43.587502</td>
      <td>41.811283</td>
      <td>82271200</td>
    </tr>
  </tbody>
</table>
</div>




```python
appl_df['Open'].plot(title="Apple's stock price")
```




    <AxesSubplot:title={'center':"Apple's stock price"}, xlabel='Date'>




    
![png](output_files/output_2_1.png)
    



```python
appl_df['Open']=appl_df['Open'].pct_change()
appl_df['Open'].plot(title="Apple's stock return")
```




    <AxesSubplot:title={'center':"Apple's stock return"}, xlabel='Date'>




    
![png](output_files/output_3_1.png)
    



```python
sc = StandardScaler()
```


```python
def preproc( data, lag, ratio):
    data=data.dropna().iloc[:, 0:1]
    Dates=data.index.unique()
    data.iloc[:, 0] = sc.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))
    for s in range(1, lag):
        data['shift_{}'.format(s)] = data.iloc[:, 0].shift(s)
    X_data = data.dropna().drop(['Open'], axis=1)
    y_data = data.dropna()[['Open']]
    index=int(round(len(X_data)*ratio))
    X_data_train=X_data.iloc[:index,:]
    X_data_test =X_data.iloc[index+1:,:]
    y_data_train=y_data.iloc[:index,:]
    y_data_test =y_data.iloc[index+1:,:]
    return X_data_train,X_data_test,y_data_train,y_data_test,Dates;
```


```python
a,b,c,d,e=preproc(appl_df, 25, 0.90)
```


```python
a.shape
```




    (429, 24)




```python
spy_df = yf.download('SPY', 
                      start='2018-01-01', 
                      end='2019-12-31', 
                      progress=False)
spy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02</th>
      <td>267.839996</td>
      <td>268.809998</td>
      <td>267.399994</td>
      <td>268.769989</td>
      <td>253.283142</td>
      <td>86655700</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>268.959991</td>
      <td>270.640015</td>
      <td>268.959991</td>
      <td>270.470001</td>
      <td>254.885162</td>
      <td>90070400</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>271.200012</td>
      <td>272.160004</td>
      <td>270.540009</td>
      <td>271.609985</td>
      <td>255.959488</td>
      <td>80636400</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>272.510010</td>
      <td>273.559998</td>
      <td>271.950012</td>
      <td>273.420013</td>
      <td>257.665283</td>
      <td>83524000</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>273.309998</td>
      <td>274.100006</td>
      <td>272.980011</td>
      <td>273.920013</td>
      <td>258.136414</td>
      <td>57319200</td>
    </tr>
  </tbody>
</table>
</div>




```python
spy_df['Open']=spy_df['Open'].pct_change()
spy_df['Open'].plot(title="Apple's stock return")
```




    <AxesSubplot:title={'center':"Apple's stock return"}, xlabel='Date'>




    
![png](output_files/output_9_1.png)
    



```python
def preproc2( data1, data2, lag, ratio):
    common_dates=list(set(data1.index) & set(data2.index))
    data1=data1[data1.index.isin(common_dates)]
    data2=data2[data2.index.isin(common_dates)]
    X1=preproc(data1, lag, ratio)
    X2=preproc(data2, lag, ratio)
    return X1,X2;
```


```python
dataLSTM=preproc2( spy_df, appl_df, 25, 0.90)
```


```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.backend as K
from keras.callbacks import EarlyStopping
```


```python
a = a.values
b= b.values

c = c.values
d = d.values


```


```python
X_train_t = a.reshape(a.shape[0], 1, 24)
X_test_t = b.reshape(b.shape[0], 1, 24)
```


```python
X_test_t.shape
```




    (47, 1, 24)




```python
K.clear_session()
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
model = Sequential()
model.add(LSTM(12, input_shape=(1, 24), return_sequences=True))
model.add(LSTM(6))
model.add(Dense(6))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```


```python
model.fit(X_train_t, c,
          epochs=100, batch_size=1, verbose=1,
          callbacks=[early_stop])
```

    Epoch 1/100
    429/429 [==============================] - 2s 1ms/step - loss: 1.3043
    Epoch 2/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.9467
    Epoch 3/100
    429/429 [==============================] - 0s 1ms/step - loss: 1.1288
    Epoch 4/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.9817
    Epoch 5/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.9464
    Epoch 6/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.9531
    Epoch 7/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.7813
    Epoch 8/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.7439
    Epoch 9/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.5139
    Epoch 10/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.4443
    Epoch 11/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.3916
    Epoch 12/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.3264
    Epoch 13/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.2933
    Epoch 14/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.2663
    Epoch 15/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.2493
    Epoch 16/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.2524
    Epoch 17/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.2164
    Epoch 18/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.1662
    Epoch 19/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.1675
    Epoch 20/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.1529
    Epoch 21/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.1656
    Epoch 22/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.1398
    Epoch 23/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.1287
    Epoch 24/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.1143
    Epoch 25/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.0988
    Epoch 26/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.1050
    Epoch 27/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.0918
    Epoch 28/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.0781
    Epoch 29/100
    429/429 [==============================] - 0s 1ms/step - loss: 0.0728
    Epoch 00029: early stopping





    <tensorflow.python.keras.callbacks.History at 0x7fca98b138b0>




```python
ypredr=[]
st=X_test_t[0].reshape(1, 1, 24)
tmp=st
ptmp=st
val=model.predict(st)
ypredr.append(val.tolist()[0])
for i in range(1, X_test_t.shape[0]):
    tmp=np.append(val, tmp[0,0, 0:-1])
    tmp=tmp.reshape(1, 1, 24)
    ptmp=np.vstack((ptmp,tmp))
    val=model.predict(tmp)
    ypredr.append(val.tolist()[0])
```


```python
plt.plot(ypredr,color="green", label = "Rolling prediction")
plt.legend()
plt.show()
```


    
![png](output_files/output_19_0.png)
    



```python
y_pred = model.predict(X_test_t)
plt.plot(d, label = "Real data")
plt.plot(y_pred, label = "One point prediction")
plt.plot(ypredr, label = "Rolling prediction")
plt.legend()
plt.show()
```


    
![png](output_files/output_20_0.png)
    



```python
Aa = dataLSTM[0][0].values
Ab = dataLSTM[0][1].values

Ac = dataLSTM[0][2].values
Ad = dataLSTM[0][3].values
X_train_A = Aa.reshape(Aa.shape[0], 1, 24)
X_test_A = Ab.reshape(Ab.shape[0], 1, 24)
```


```python
Sa = dataLSTM[1][0].values
Sb = dataLSTM[1][1].values

Sc = dataLSTM[1][2].values
Sd = dataLSTM[1][3].values
X_train_S = Sa.reshape(Sa.shape[0], 1, 24)
X_test_S = Sb.reshape(Sb.shape[0], 1, 24)
```


```python
from keras.layers import concatenate
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.models import Input, Model
from keras.layers import Dense


```


```python
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
input1 = Input(shape=(1,24)) # for the three columns of dat_train
x1 = LSTM(6)(input1)

input2 = Input(shape=(1,24))
x2 = LSTM(6)(input2)

con = concatenate(inputs = [x1,x2] ) # merge in metadata
x3 = Dense(50)(con)
x3 = Dropout(0.3)(x3)
output = Dense(1, activation='sigmoid')(x3)
n_net = Model(inputs=[input1, input2], outputs=output)
n_net.compile(loss='mean_squared_error', optimizer='adam')
```


```python
n_net.fit(x=[X_train_A, X_train_S], y=Ac, epochs=100, batch_size=1, verbose=1,
          callbacks=[early_stop])
```

    Epoch 1/100
    429/429 [==============================] - 0s 832us/step - loss: 0.7942
    Epoch 2/100
    429/429 [==============================] - 0s 808us/step - loss: 0.7825
    Epoch 3/100
    429/429 [==============================] - 0s 819us/step - loss: 0.7702
    Epoch 4/100
    429/429 [==============================] - 0s 792us/step - loss: 0.7589
    Epoch 5/100
    429/429 [==============================] - 0s 786us/step - loss: 0.7517
    Epoch 6/100
    429/429 [==============================] - 0s 787us/step - loss: 0.7463
    Epoch 7/100
    429/429 [==============================] - 0s 817us/step - loss: 0.7421
    Epoch 8/100
    429/429 [==============================] - 0s 805us/step - loss: 0.7372
    Epoch 9/100
    429/429 [==============================] - 0s 798us/step - loss: 0.7301
    Epoch 10/100
    429/429 [==============================] - 0s 792us/step - loss: 0.7260
    Epoch 11/100
    429/429 [==============================] - 0s 821us/step - loss: 0.7236
    Epoch 12/100
    429/429 [==============================] - 0s 820us/step - loss: 0.7156
    Epoch 13/100
    429/429 [==============================] - 0s 833us/step - loss: 0.7134
    Epoch 14/100
    429/429 [==============================] - 0s 802us/step - loss: 0.7143
    Epoch 00014: early stopping





    <tensorflow.python.keras.callbacks.History at 0x7fca98b89880>




```python
y_pred = n_net.predict([X_test_A,X_test_S])
plt.plot(Ad, label = "Real data")
plt.plot(y_pred, label = "One point prediction")
plt.legend()
plt.show()
```


    
![png](output_files/output_26_0.png)
    



```python
ypredr=[]
st=X_test_A[0].reshape(1, 1, 24)
sst=X_test_S[0].reshape(1, 1, 24)
tmp=st
ptmp=st
val=n_net.predict([tmp,sst])
ypredr.append(val.tolist()[0])
for i in range(1, X_test_t.shape[0]):
    tmp=np.append(val, tmp[0,0, 0:-1])
    tmp=tmp.reshape(1, 1, 24)
    sst=X_test_S[i].reshape(1, 1, 24)
    ptmp=np.vstack((ptmp,tmp))
    val=n_net.predict([tmp,sst])
    ypredr.append(val.tolist()[0])
```


```python
plt.plot(ypredr, color="green", label = "Rolling prediction")
plt.legend()
plt.show()
```


    
![png](output_files/output_28_0.png)
    



```python
y_pred = n_net.predict([X_test_A,X_test_S])
plt.plot(Ad, label = "Real data")
plt.plot(y_pred, label = "One point prediction")
plt.plot(ypredr, label = "Rolling prediction")
plt.legend()
plt.show()
```


    
![png](output_files/output_29_0.png)
    



```python

```
