#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import urllib
import pandas as pd
import numpy as np


# In[2]:


iris = datasets.load_iris()


# In[3]:


print(iris.DESCR)
print(iris.data[0])
print(iris.data.shape)
print(iris.feature_names)
print(iris.target)
print(iris.target.shape)
print(iris.target_names)
print(type(iris.data))


# In[4]:


colors = list()
palette = {0: 'red', 1: 'green', 2: 'blue'}


# In[5]:


for c in np.nditer(iris.target):
    colors.append(palette[int(c)])
    dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[9]:


sc = pd.plotting.scatter_matrix(dataframe, alpha=0.3, figsize=(10,10),
                      diagonal='hist', color=colors, marker='o', grid=True)


# In[11]:


url = "http://aima.cs.berkeley.edu/data/iris.csv"
set1 = urllib.request.Request(url)
iris_p = urllib.request.urlopen(set1)
iris_other = pd.read_csv(iris_p, sep=',', decimal='.', header=None, 
                          names=['sepal_lenght', 'sepal_width', 'petal_length', 'petal_width', 'target'])
iris_other.head()


# In[13]:


iris_other.tail()


# In[15]:


iris_other.columns # returns not a list but index of pandas libraries


# In[25]:


Y = iris_other['target']
Y


# In[23]:


X = iris_other[['sepal_lenght', 'sepal_width']]
X


# In[24]:


print(X.shape)


# In[26]:


print(Y.shape)

