# -*- coding: utf-8 -*-
"""
Created on Sun May 15 01:44:57 2020

@author: Dash
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

# Retrieving data
data = pd.read_csv(r'supplement1\ms_toy.txt', delimiter = '\s+');

# 1.Create a dataset containing only the three control samples 
data_cols = ['ctrl.1', 'ctrl.2', 'ctrl.3'];
dataset = data[data_cols];

# 2.Remove any row where all three samples have missing values
dataset.dropna(how = 'all',inplace = True);
dataset.reset_index(drop=True, inplace=True);

# Saving ctrl.1 data for plot
data_1 = dataset['ctrl.1'];
s = data_1[pd.isnull(data_1)];

# Performing KNN imputation using sklearn
imputer = KNNImputer(n_neighbors = 3);
dataset_i = imputer.fit_transform(dataset);

data_2 = dataset_i[:,0];
data_imputed = [];

for row in s.index.tolist():
    data_imputed.append(data_2[row]);
    
# Plot imputed data vs overall data
plt.figure(figsize=(18,12));   
plt.title('Data imputation using KNN imputer');
plt.grid(axis = 'both', color='w', linestyle='-', linewidth= 0.3);
p1 = plt.hist(x = data_1, color = 'blue', edgecolor='black',bins = 100, label='overall data');
p2 = plt.hist(x = data_imputed, color = 'red', edgecolor='black',bins = 25, label='imputed data');
plt.legend();
plt.xlabel('Intensity');
plt.ylabel('Count');
plt.show()


# For comparison
print(np.mean(data_1))
print(np.std(data_1))
print(np.mean(data_imputed))
print(np.std(data_imputed))