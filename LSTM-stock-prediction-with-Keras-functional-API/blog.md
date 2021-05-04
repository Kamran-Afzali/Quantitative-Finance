Introduction


 Time-series & forecasting models
Traditionally most machine learning (ML) models use as input features some observations (samples / examples) but there is no time dimension in the data.
Time-series forecasting models are the models that are capable to predictfuture values based on previously observed values. Time-series forecasting is widely used for non-stationary data. Non-stationary data are called the data whose statistical properties e.g. the mean and standard deviation are not constant over time but instead, these metrics vary over time.
These non-stationary input data (used as input to these models) are usually called time-series. Some examples of time-series include the temperature values over time, stock price over time, price of a house over time etc. So, the input is a signal (time-series) that is defined by observations taken sequentially in time.

Observation: Time-series data is recorded on a discrete time scale.
Disclaimer (before we move on): There have been attempts to predict stock prices using time series analysis algorithms, though they still cannot be used to place bets in the real market. This is just a tutorial article that does not intent in any way to “direct” people into buying stocks.
2. The LSTM model
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (e.g. images), but also entire sequences of data (such as speech or video inputs).
LSTM models are able to store information over a period of time.
In order words, they have a memory capacity. Remember that LSTM stands for Long Short-Term Memory Model.
This characteristic is extremely useful when we deal with Time-Series or Sequential Data. When using an LSTM model we are free and able to decide what information will be stored and what discarded. We do that using the “gates”. The deep understanding of the LSTM is outside the scope of this post but if you are interested in learning more, have a look at the references at the end of this post.
3. Getting the stock price history data
Thanks to Yahoo finance we can get the data for free. Use the following link to get the stock price history of TESLA:
 
In this tutorial, we’ll build a Python deep learning model that will predict the future behavior of stock prices. We assume that the reader is familiar with the concepts of deep learning in Python, especially Long Short-Term Memory.
While predicting the actual price of a stock is an uphill climb, we can build a model that will predict whether the price will go up or down. The data and notebook used for this tutorial can be found here. It’s important to note that there are always other factors that affect the prices of stocks, such as the political atmosphere and the market. However, we won’t focus on those factors for this tutorial.
 
Introduction
 
LSTMs are very powerful in sequence prediction problems because they’re able to store past information. This is important in our case because the previous price of a stock is crucial in predicting its future price.
We’ll kick of by importing NumPy for scientific computation, Matplotlib for plotting graphs, and Pandas to aide in loading and manipulating our datasets.

Loading the Dataset
 
The next step is to load in our training dataset and select the Open and Highcolumns that we’ll use in our modeling.

The Open column is the starting price while the Close column is the final price of a stock on a particular trading day. The High and Low columns represent the highest and lowest prices for a certain day.
Feature Scaling
From previous experience with deep learning models, we know that we have to scale our data for optimal performance. In our case, we’ll use Scikit- Learn’s MinMaxScaler and scale our dataset to numbers between zero and one.

Building the LSTM
 
In order to build the LSTM, we need to import a couple of modules from Keras:
Sequential for initializing the neural network
Dense for adding a densely connected neural network layer
LSTM for adding the Long Short-Term Memory layer
Dropout for adding dropout layers that prevent overfitting

We add the LSTM layer and later add a few Dropout layers to prevent overfitting. We add the LSTM layer with the following arguments:

50 units which is the dimensionality of the output space
return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence
input_shape as the shape of our training set.
When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped. Thereafter, we add the Dense layer that specifies the output of 1 unit. After this, we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error. This will compute the mean of the squared errors. Next, we fit the model to run on 100 epochs with a batch size of 32. Keep in mind that, depending on the specs of your computer, this might take a few minutes to finish running.

Predicting Future Stock using the Test Set
 
First we need to import the test set that we’ll use to make our predictions on.

In order to predict future stock prices we need to do a couple of things after loading in the test set:
1.	Merge the training set and the test set on the 0 axis.
2.	Set the time step as 60 (as seen previously)
3.	Use MinMaxScaler to transform the new dataset
4.	Reshape the dataset as done previously
After making the predictions we use inverse_transform to get back the stock prices in normal readable format.


Plotting the Results
 
Finally, we use Matplotlib to visualize the result of the predicted stock price and the real stock price.

From the plot we can see that the real stock price went up while our model also predicted that the price of the stock will go up. This clearly shows how powerful LSTMs are for analyzing time series and sequential data.

 

Conclusion
 
There are a couple of other techniques of predicting stock prices such as moving averages, linear regression, K-Nearest Neighbours, ARIMA and Prophet. These are techniques that one can test on their own and compare their performance with the Keras LSTM. If you wish to learn more about Keras and deep learning you can find my articles on that here and here.


https://www.datacamp.com/community/tutorials/lstm-python-stock-market


Splitting Data into a Training set and a Test set

You will use the mid price calculated by taking the average of the highest and lowest recorded prices on a day. Now you can split the training data and test data. The training data will be the first 11,000 data points of the time series and rest will be test data.

Normalizing the Data

Now you need to define a scaler to normalize the data. MinMaxScalar scales all the data to be in the region of 0 and 1. You can also reshape the training and test data to be in the shape [data_size, num_features].

Due to the observation you made earlier, that is, different time periods of data have different value ranges, you normalize the data by splitting the full series into windows. If you don't do this, the earlier data will be close to 0 and will not add much value to the learning process. Here you choose a window size of 2500.
Tip: when choosing the window size make sure it's not too small, because when you perform windowed-normalization, it can introduce a break at the very end of each window, as each window is normalized independently.
In this example, 4 data points will be affected by this. But given you have 11,000 data points, 4 points will not cause any issue
