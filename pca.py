# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:34:04 2020

@author: Dash
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Retrieving data
data = pd.read_csv(r'supplement1\pca_toy.txt', delimiter = '\s+');

# Gathering features : all columns is an exception
feature_cols = ['a', 'b', 'c', 'd'];

X = data[feature_cols];

# Standardizing the features
x = StandardScaler().fit_transform(X);

# configuring no. of PCA components
pca = PCA(n_components = 2);

principalComponents = pca.fit_transform(x);

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2']);

# Plotting ScatterPlot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('PCA with 2 components', fontsize = 20)

plt.scatter(principalDf['PC1'], principalDf['PC2']);
plt.show()
ax.grid()

# To find which variables have more impact on which PCs
print(abs( pca.components_ ))

# Deriving PCs variance coverage
var = pca.explained_variance_ratio_
