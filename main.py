#from dotenv import load_dotenv


from handlers.lemon import LemonMarketsAPI



#load_dotenv()


import pandas as pd
import yfinance as yf
import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from model import *

# show data for different tickers
#isin = os.environ.get("isin")
#start = os.environ.get("startDate")
#

start = pd.to_datetime('2018-08-01')
isin = ['ETH-USD']
testShare = 0.05

#Model
modelMetric = 'MSE'

epochs = 20
batch_size = 16


data = yf.download(isin, start=start, end=datetime.date.today())
print(data.head())

# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#data = scaler.fit_transform(data)

# split into train and test
Y = data['Adj Close']
X = data.drop('Adj Close', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testShare)

# train model
model = LongShortTermMemory(X_train)

callback = get_callback(model)

#callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=modelMetric)
model.fit(X_train, Y_train, epochs, batch_size, validation_data=(X_test, Y_test),
                        callbacks=callback)

#use XGBOOST model


