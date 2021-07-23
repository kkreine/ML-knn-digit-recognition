#!/usr/bin/env python
# coding: utf-8

# In[23]:


# Import libraries:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[24]:


data = pd.read_csv('digits_recognition.csv', header=None)


# In[25]:


#check for missing values
data.isna().sum()


# In[26]:


# Extract the feature vector and the output label
# Note the output label is at column 0
# the feature vector starts from column 1 to the last column
X = data.values[:, 1:]  
y = data.values[:, 0]
X.shape


# In[27]:


_, axarr = plt.subplots(5,5,figsize=(5,5))
for i in range(5):
    for j in range(5):
        ## get an int random number from 5000 examples
        r = np.random.randint(X.shape[0])
        ## get that image from X
        XA = X[r].reshape((28, 28)) 
        ## show in sub-figure (i, j)
        axarr[i,j].imshow(XA, cmap='gray')  
        axarr[i,j].axis('off') ## turn off axis...
plt.show()


# In[28]:


# Scale the feature vector by the range
X.min()
X.max()
X = X/(X.max() - X.min())


# In[29]:


# Split the feature vector and output label into X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 47)
print(x_train.shape)
print(x_test.shape)
print('x_train.shape[0]+x_test.shape[0]=', x_train.shape[0] + x_test.shape[0])


# In[30]:


# Use knn to find best fit
from sklearn.neighbors import KNeighborsRegressor
test_scores = []
train_scores = []
K = []
for k in range(1, 12):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train, y_train)
    K.append(k)
    test_scores.append(knn.score(x_test, y_test))
    train_scores.append(knn.score(x_train, y_train))
    print('K=', K[k-1], ' test_score=', test_scores[k-1], '  train_test=', train_scores[k-1])

m = max(test_scores)
i = test_scores.index(m)
print('max accuracy ', m, '  train score: ', train_scores[i], '   for K=', K[i])


# In[21]:


from sklearn.model_selection import GridSearchCV   # Grid search with cross validation

param = {'n_neighbors': list(range(1, 50))}

gs = GridSearchCV(KNeighborsRegressor(), param, cv=4)
result = gs.fit(x_train, y_train)

print(result.best_score_)
print(result.best_estimator_)
print(gs.score(x_test, y_test))

