#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
filename = 'Zoom.csv' 
if '_MEIPASS2' in os.environ:
    filename = os.path.join(os.environ['_MEIPASS2'], filename)
fd = open(filename, 'rb')
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
dataset_train  = pd.read_csv('Zoom.csv')
# print(dataset_train.head())

# In[3]:


training_set = dataset_train.iloc[:, 1:2].values


# In[4]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[70]:
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense


# In[86]:


from numpy import array
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
n_steps = 3
X, y = split_sequence(training_set, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))

# In[87]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle=False)


# In[88]:


model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=200, verbose=0)


# In[89]:


# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_percentage_error
# print(mean_squared_error(y_test, model.predict(X_test), squared=False))
# print(mean_absolute_percentage_error(y_test, model.predict(X_test)))


# In[60]:
from numpy import array
# userInput = array([93.4,91.36,88.68]) #input harga opening 3 hari telakhir
# userInput = userInput.reshape((1, n_steps, 1))


# # In[61]:


# pred = model.predict(userInput)
# print(pred) #harga closing besoknya


# In[ ]:

def predict_price(first_opening, second_opening, third_opening):
	userInput = array([first_opening,second_opening,third_opening]).astype('float32')
	userInput = userInput.reshape((1, n_steps, 1))
	pred = model.predict(userInput)
	return pred
	
def corr():
	plt.figure()
	x=dataset_train.corr()
	mask = np.triu(np.ones_like(x, dtype=np.bool))
	sns.heatmap(x,annot=True,mask=mask,vmin=-1,vmax=1)
	plt.title("Zoom Dataset Correlation Heatmap")
	plt.show()
	

