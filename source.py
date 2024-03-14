import numpy as np
import pandas as pd
import matplotlib 
import theano 
import tensorflow 
import keras 

dataset = pd.read_csv('D:\\Agriculture\SoilDataset.csv') 
X = dataset.iloc[:, 3:32].values
y = dataset.iloc[:, 32].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_0 = LabelEncoder() 
X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0]) 

labelencoder_X_1 = LabelEncoder() 
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) 

labelencoder_X_2 = LabelEncoder() 
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) 

# Repeat for other columns

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 

import keras 
from keras.models import Sequential
from keras.layers import Dense 

classifier = Sequential() 
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 1422))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) 
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100) 

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
print(cm) 

from sklearn.metrics import accuracy_score 
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)))
