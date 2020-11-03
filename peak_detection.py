# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 18:06:33 2020

@author: Dash
"""


#import packages
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#open a read file
file = open('yeast_cell_cycle.txt')
f = file.read()

#the heading, but we are said to ignore this
a = f.split('\n')[0].split('\t')

#relevant strings to df
swi = re.findall(r'YER111C.+', f)
mcm = re.findall(r'YMR043W.+', f)
ace = re.findall(r'YLR131C.+', f)
al = swi + mcm + ace
al = pd.DataFrame(al)
al = pd.DataFrame(al[al.columns[0]] .str.split('\t',100).tolist())
#al = al.drop(al.columns[10], axis = 1)
#al.columns = a
al = al.transpose()
al = al.drop([0], axis = 0)
al.columns = ['SWI4', 'MCM1', 'ACE2']
al.replace('', 0)
al = al.apply(lambda x: pd.to_numeric(x))
al = al.fillna(0)


#Plot the data
sns.lineplot(data = al)
plt.savefig('Expression of 3 genes')
# It can be seen from the plot that expression of all three genes varies significantly during the cell cycle.

# In order to mark significant peaks, we use Watershed algorithm:
def Watershed(gene, adj_threshold):
    data = pd.DataFrame(al[gene])
    data['label'] = 0   # make a column for labels
    label = 0           #set the label to 0
    my_list = [i for i in range(1,83)]                                #A list with indexes we have not yet labeled
    while data[data.label == 0].shape[0] > 0:
        if data[gene][my_list].max() > abs(data[gene][my_list].min()): 
            wl = data[gene][my_list].max()                            #set the water level to the max or min
        else:
            wl = data[gene][my_list].min()
        wl_index = data[gene].index[data[gene] == wl].tolist()        #find the indexes of this value
        for ind in wl_index:
            if (ind - adj_threshold) <= 0 and (ind + adj_threshold+1) < 82:  #if the index is too close to 0
                l = [i for i in range(1, ind + adj_threshold+1)]

            elif ((ind + adj_threshold+1) > 82) and (ind - adj_threshold) > 0: #if the index is too close to 82
                l = [i for i in range(ind - adj_threshold, 82)]

            elif (ind - adj_threshold) < 0 and ((ind + adj_threshold + 1) > 82): #if the adj threshold is huge
                l = data[gene].index[data['label'] == 0].tolist()

            else:                                                                #if everything is fine
                l = [i for i in range(ind - adj_threshold, ind)] + [i for i in range(ind+1, ind + adj_threshold + 1)]
        #nearest values from nearest indexes:
            nearest = data['label'][l].tolist()
            r = [i for i in nearest if i > 0] #list of close points, which are already labeled
            if len(r) > 0:
                data['label'][ind] = r[0]     #if the exist, copy label of the leftmost (can be set differently)
            else:
                label += 1
                data['label'][ind] = label    #if there are none make a new one
            my_list.remove(ind)
            
    plt.figure(figsize=(12,8))               #visualise
    sns.pointplot(x = data.index, y=data[gene], palette="tab10", hue = data.label, legend = False)
    plt.legend(loc='upper right')
    name = gene + ' ' + str(adj_threshold)
    plt.savefig(name)


# **Adjacement threshold = 1**
Watershed('SWI4', 1)

# **Adjacement threshold = 2**
Watershed('SWI4', 2)

# **Adjacement threshold = 3**
Watershed('SWI4', 3)

# **Adjacement threshold = 4**
Watershed('SWI4', 4)

# **Adjacement threshhold = 3 seems to be the best one!**

# Here is exactly the same function as before, but it returns the table instead of the graph
def Watershed_data(gene, adj_threshold):
    data = pd.DataFrame(al[gene])
    data['label'] = 0   # make a column for labels
    label = 0           #set the label to 0
    my_list = [i for i in range(1,83)]                                #A list with indexes we have not yet labeled
    while data[data.label == 0].shape[0] > 0:
        if data[gene][my_list].max() > abs(data[gene][my_list].min()): 
            wl = data[gene][my_list].max()                            #set the water level to the max or min
        else:
            wl = data[gene][my_list].min()
        wl_index = data[gene].index[data[gene] == wl].tolist()        #find the indexes of this value
        for ind in wl_index:
            if (ind - adj_threshold) <= 0 and (ind + adj_threshold+1) < 82:  #if the index is too close to 0
                l = [i for i in range(1, ind + adj_threshold+1)]

            elif ((ind + adj_threshold+1) > 82) and (ind - adj_threshold) > 0: #if the index is too close to 82
                l = [i for i in range(ind - adj_threshold, 82)]

            elif (ind - adj_threshold) < 0 and ((ind + adj_threshold + 1) > 82): #if the adj threshold is huge
                l = data[gene].index[data['label'] == 0].tolist()

            else:                                                                #if everything is fine
                l = [i for i in range(ind - adj_threshold, ind)] + [i for i in range(ind+1, ind + adj_threshold + 1)]
        #nearest values from nearest indexes:
            nearest = data['label'][l].tolist()
            r = [i for i in nearest if i > 0] #list of close points, which are already labeled
            if len(r) > 0:
                data['label'][ind] = r[0]     #if the exist, copy label of the leftmost (can be set differently)
            else:
                label += 1
                data['label'][ind] = label    #if there are none make a new one
            my_list.remove(ind)
    return data      
    
#Compute the labels for each gene data, Adj_threshold = 3
mcm = Watershed_data('MCM1', 3)
ace = Watershed_data('ACE2', 3)
swi = Watershed_data('SWI4', 3)

#MCM1
plt.figure(figsize=(12,8))               #visualise
sns.lineplot(x = mcm.index, y=mcm['MCM1'], color = 'grey')
#sns.lineplot(x = ace.index, y=ace['ACE2'])
sns.scatterplot(x = mcm.index, y=mcm['MCM1'], palette="Set1", hue = mcm.label)
#sns.scatterplot(x = ace.index, y=ace['ACE2'], palette="tab10", hue = ace.label, legend = False)
plt.legend(loc='upper right')
plt.savefig('MCM')

#ACE2
plt.figure(figsize=(12,8))               
sns.lineplot(x = ace.index, y=ace['ACE2'], color = 'grey')
sns.scatterplot(x = ace.index, y=ace['ACE2'], palette="Set1", hue = ace.label)
plt.legend(loc='upper right')
plt.savefig('ACE')

#SWI4 
plt.figure(figsize=(12,8))               
sns.lineplot(x = swi.index, y=swi['SWI4'], color = 'grey')
sns.scatterplot(x = swi.index, y=swi['SWI4'], palette="Set1", hue = ace.label)
plt.legend(loc='upper right')
plt.savefig('SWI4')
