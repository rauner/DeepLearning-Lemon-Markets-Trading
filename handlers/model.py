#-------------------------------------------------------------------------------
# Name:        model
# Purpose:
#
# Author:      rauner
#
# Created:     16/03/2022
# Copyright:   (c) rauner 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dropout, Dense, LSTM


def compile_and_fit(model, window, patience=2, MAX_EPOCHS = 20):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

def conv_model(filters = 32, kernel_size = 10, activation = 'relu'):
  model = tf.keras.Sequential([
  tf.keras.layers.Conv1D(filters=filters,
                           kernel_size=(kernel_size,),
                           activation=activation),
  tf.keras.layers.Dense(units=32, activation=activation),
  tf.keras.layers.Dense(units=1),])

  return model
