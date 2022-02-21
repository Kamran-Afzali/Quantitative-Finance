

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
                      start='2019-01-01', 
                      end='2020-12-31', 
                      progress=False)

```


```python
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
      <th>2019-01-02</th>
      <td>38.722500</td>
      <td>39.712502</td>
      <td>38.557499</td>
      <td>39.480000</td>
      <td>38.277515</td>
      <td>148158800</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>35.994999</td>
      <td>36.430000</td>
      <td>35.500000</td>
      <td>35.547501</td>
      <td>34.464798</td>
      <td>365248800</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>36.132500</td>
      <td>37.137501</td>
      <td>35.950001</td>
      <td>37.064999</td>
      <td>35.936085</td>
      <td>234428400</td>
    </tr>
    <tr>
      <th>2019-01-07</th>
      <td>37.174999</td>
      <td>37.207500</td>
      <td>36.474998</td>
      <td>36.982498</td>
      <td>35.856094</td>
      <td>219111200</td>
    </tr>
    <tr>
      <th>2019-01-08</th>
      <td>37.389999</td>
      <td>37.955002</td>
      <td>37.130001</td>
      <td>37.687500</td>
      <td>36.539616</td>
      <td>164101200</td>
    </tr>
  </tbody>
</table>
</div>




```python
appl_df['Open'].plot(title="Apple's stock price")
```




    <AxesSubplot:title={'center':"Apple's stock price"}, xlabel='Date'>




    
![png](/images/tuner_3_1.png)
    



```python
appl_df['Open']=appl_df['Open'].pct_change()
appl_df['Open'].plot(title="Apple's stock return")
```




    <AxesSubplot:title={'center':"Apple's stock return"}, xlabel='Date'>




    
![png](/images/tuner_4_1.png)
    



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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.backend as K
from keras.callbacks import EarlyStopping
import keras_tuner as kt
from tensorflow.keras.layers import Dropout
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
```


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
    431/431 [==============================] - 2s 973us/step - loss: 0.9624
    Epoch 2/100
    431/431 [==============================] - 0s 947us/step - loss: 1.0980
    Epoch 3/100
    431/431 [==============================] - 0s 947us/step - loss: 0.9530
    Epoch 4/100
    431/431 [==============================] - 0s 945us/step - loss: 0.8867
    Epoch 5/100
    431/431 [==============================] - 0s 943us/step - loss: 0.8433
    Epoch 6/100
    431/431 [==============================] - 0s 942us/step - loss: 0.5886
    Epoch 7/100
    431/431 [==============================] - 0s 956us/step - loss: 0.6192
    Epoch 8/100
    431/431 [==============================] - 0s 973us/step - loss: 0.5257
    Epoch 9/100
    431/431 [==============================] - 0s 957us/step - loss: 0.4120
    Epoch 10/100
    431/431 [==============================] - 0s 946us/step - loss: 0.3625
    Epoch 11/100
    431/431 [==============================] - 0s 944us/step - loss: 0.3114
    Epoch 12/100
    431/431 [==============================] - 0s 943us/step - loss: 0.3296
    Epoch 13/100
    431/431 [==============================] - 0s 944us/step - loss: 0.2298
    Epoch 14/100
    431/431 [==============================] - 0s 945us/step - loss: 0.2337
    Epoch 15/100
    431/431 [==============================] - 0s 945us/step - loss: 0.2314
    Epoch 16/100
    431/431 [==============================] - 0s 942us/step - loss: 0.2489
    Epoch 17/100
    431/431 [==============================] - 0s 940us/step - loss: 0.2131
    Epoch 18/100
    431/431 [==============================] - 0s 943us/step - loss: 0.1688
    Epoch 19/100
    431/431 [==============================] - 0s 947us/step - loss: 0.1759
    Epoch 20/100
    431/431 [==============================] - 0s 974us/step - loss: 0.1767
    Epoch 21/100
    431/431 [==============================] - 0s 1ms/step - loss: 0.1603
    Epoch 22/100
    431/431 [==============================] - 1s 1ms/step - loss: 0.1526
    Epoch 23/100
    431/431 [==============================] - 0s 1ms/step - loss: 0.1666
    Epoch 24/100
    431/431 [==============================] - 0s 945us/step - loss: 0.1575
    Epoch 00024: early stopping





    <tensorflow.python.keras.callbacks.History at 0x7fe740a444c0>




```python

def build_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=128,step=32),return_sequences=True, input_shape=(1,24)))
    for i in range(hp.Int('n_layers', 1, 10)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=128,step=32),return_sequences=True))
    model.add(LSTM(6))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(6))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
    return model
```


```python
tuner= kt.RandomSearch(
        build_model,
        objective='mse',
        max_trials=10,
        executions_per_trial=3
        )
```

    INFO:tensorflow:Reloading Oracle from existing project ./untitled_project/oracle.json
    INFO:tensorflow:Reloading Tuner from ./untitled_project/tuner0.json



```python
tuner.search(
        x=X_train_t,
        y=c,
        epochs=20,
        batch_size=128,
        validation_data=(X_test_t,d),
)
```

    Trial 10 Complete [00h 00m 17s]
    mse: 0.8778028885523478
    
    Best mse So Far: 0.8115118543306986
    Total elapsed time: 00h 02m 14s
    INFO:tensorflow:Oracle triggered exit



```python
best_model = tuner.get_best_models(num_models=1)[0]
```


```python
best_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 1, 64)             22784     
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 1, 64)             33024     
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 1, 96)             61824     
    _________________________________________________________________
    lstm_3 (LSTM)                (None, 6)                 2472      
    _________________________________________________________________
    dropout (Dropout)            (None, 6)                 0         
    _________________________________________________________________
    dense (Dense)                (None, 6)                 42        
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 6)                 0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 7         
    =================================================================
    Total params: 120,153
    Trainable params: 120,153
    Non-trainable params: 0
    _________________________________________________________________



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


    
![png](/images/tuner_19_0.png)
    



```python
y_pred = model.predict(X_test_t)
plt.plot(d, label = "Real data")
plt.plot(y_pred, label = "One point prediction")
plt.plot(ypredr, label = "Rolling prediction")
plt.legend()
plt.show()
```


    
![png](/images/tuner_20_0.png)
    

