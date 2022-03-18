#from dotenv import load_dotenv


from handlers.lemon import LemonMarketsAPI
#load_dotenv()
import pandas as pd
import numpy as np


import yfinance as yf
from datetime import datetime, timedelta, date


from handlers import model
from handlers import helpers

# show data for different tickers
#isin = os.environ.get("isin")
#start = os.environ.get("startDate")
#


isin = ['ETH-USD']
testShare = 0.05

interval = 2 #time steps in minutes

# intraday data only available from yf for the last 60 days
data = yf.download(isin, start=date.today() -  timedelta(days=59), end=date.today(), interval= str(interval) + 'm')
print(data.head())

# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#data = scaler.fit_transform(data)


# include days and weeks as cos features
data = data.reset_index(level=0)
seconds = data['Datetime'].map(pd.Timestamp.timestamp)


day = 24*60*60/interval
week = 7*day

data['day cos'] = np.cos(seconds * (2 * np.pi / day))
data['week cos'] = np.cos(seconds * (2 * np.pi / week))
data = data.drop('Datetime', axis = 1)

# split into train and test
n = len(data)
train_df = data[0:int(n*0.7)]
val_df = data[int(n*0.7):int(n*0.9)]
test_df = data[int(n*0.9):]

num_features = data.shape[1]


# normalization, simple mean and variance
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std




# implement covnet
conv_width = 10
label_width = 1
input_width = label_width + (conv_width - 1)

shift = 1

wide_conv_window = helpers.WindowGenerator(
    input_width=input_width,
    label_width=label_width,
    train_df = train_df,
    val_df = val_df,
    test_df = test_df,
    shift=shift,
    label_columns=['Adj Close'])

filters = 32
kernel_size = 10
activation = 'relu'
conv_model = model.conv_model()

patience=2
MAX_EPOCHS = 5
history = model.compile_and_fit(conv_model, wide_conv_window)


wide_conv_window.plot(conv_model)

'''
clean, make class and so on nice
implement that its forcasts the next h with ARNN


# implement XGBOOST
# implement random tree
# implement bagging
# boosting


'''