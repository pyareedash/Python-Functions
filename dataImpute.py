# -*- coding: utf-8 -*-
"""
Created on Wed May 13 01:12:23 2020

@author: Dash
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Retrieving data
data = pd.read_csv(r'supplement1\ms_toy.txt', delimiter = '\s+');

# Function initialization
def data_impute(A):    
    # STEP 1
    d_mean = np.mean(A); # old mean
    print(d_mean)
    d_sd = np.std(A);    # old standard deviation
    print(d_sd)
    # STEP 2
    d_lq = A.quantile(0.25); # Lower quartile ~ 0.25
    data_lq = [];       # To store lower quartile values
    for dp in A:
        if dp <= d_lq:
            data_lq.append(dp);
            
    new_mean = np.mean(data_lq); # new mean
    print(new_mean)
    # STEP 3
    new_sd = d_sd*(1/3); # new standard deviation 1/3rd of old
    print(new_sd)
    # Generate new data from new distribution and imputing
    s = A[pd.isnull(A)];    
    print(len(s))
    # STEP 4
    for row, value in zip(s.index.tolist(), np.random.normal(new_mean, new_sd, len(s))):
        A[row] = value
        s[row] = value
    
    # Plot imputed data vs overall data
    plt.figure(figsize=(18,12));   
    plt.title('Data imputation using lower quartile');
    plt.grid(axis = 'both', color='w', linestyle='-', linewidth= 0.3);
    p1 = plt.hist(x = A, color = 'blue', edgecolor='black',bins = 100, label='overall data');
    p2 = plt.hist(x = s, color = 'red', edgecolor='black',bins = 25, label='imputed data');
    plt.legend();
    plt.xlabel('Intensity');
    plt.ylabel('Count');
    plt.show()


# For column = ctrl.1
A = data['ctrl.1'];

# Using function on Column 'ctrl.1'
data_impute(A);   

    

    
    