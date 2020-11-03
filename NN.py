# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:53:31 2020

@author: Pyare
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

data_1 = pd.read_csv('data_6.1.csv');


X = data_1[['class']].values;
y = data_1.iloc[:2:].values();

#data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0);


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(14,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

