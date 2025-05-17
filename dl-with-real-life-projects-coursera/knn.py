#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors


# In[4]:


data = np.genfromtxt('./6.overlap.csv', delimiter=',')

data


# In[ ]:


from matplotlib.colors import  ListedColormap

def knn_comparison(data, n_neighbors = 15):
    
    X = data[:, :2]
    y = data[:, 2]
    
    n=.02
    cmap_light = ListedColormap(["FFAAAA", "AAAAFF"])
    cmap_bold = ListedColormap(["#f0000", "#0000FF"])
    
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(X,y)
    

