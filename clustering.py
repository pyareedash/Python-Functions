# -*- coding: utf-8 -*-
"""
Created on Sat May 30 04:42:31 2020

@author: Dash
"""

import pandas as pd
import numpy as np
import math
import itertools

#(a) Parsing input file for further tasks
data_1 = pd.read_csv('methylation.csv', sep=';', thousands='.', decimal=',');
data_1.replace('.', np.nan, inplace=True);
df = data_1[data_1.columns[7:26]];

# Removing commas ',' and replacing with '.' decimal
for i in list(df):
    df[i] = (df[i].replace(',','.', regex=True)
                        .astype(float))    

df.replace(np.nan, 0, inplace=True);
blood_lineage = df.loc[:,'HSC':'Mono'];

#(b) Average methylation levels of blood lineage
print(blood_lineage.mean())

#(c)
#1 Euclidean distance
def euclidean_dist(a,b):
    return math.sqrt(sum([(x-y)**2 for x,y in zip(a,b)]));
    
# pairwise distances between HSC, CD4and TBSC
print('Pairwise distance between HSC and CD4 : ',euclidean_dist(df.HSC, df.CD4));
print('Pairwise distance between CD4 and TBSC : ',euclidean_dist(df.CD4, df.TBSC));
print('Pairwise distance between HSC and TBSC : ',euclidean_dist(df.HSC, df.TBSC));

#2 Linkage criterion
# def linkage_crit(c,d):
    # return (1/(len(c)*len(d)))*(sum([(sum([euclidean_dist(a, b) for a in c])) for b in d]));
def linkage_crit(A,B):
    sum = 0;
    for a in A:
        for b in B:
            sum += mat.loc[a,b];
        
    L = sum/(len(A)*len(B));
        
    return L
    
#3 Agglomerative clustering

linkage_g = list(df.columns);
mat = pd.DataFrame(np.zeros((len(linkage_g),len(linkage_g))), index = linkage_g, columns = linkage_g);
col_pairs = list(itertools.combinations(linkage_g, 2));

for x,y in col_pairs:
    x!=y
    mat[x][y] = euclidean_dist(df.loc[:,x],df.loc[:,y])
    mat[y][x] = euclidean_dist(df.loc[:,y],df.loc[:,x])

clusters = [[i] for i in linkage_g];
steps = 1;

while len(clusters) > 1:
    point = 0;
    p = 0;
    pairs_col = list(itertools.combinations(clusters, 2));
    linkage_new = [];
    r = -1;
    c = -1;
        
    for x in pairs_col:
        p = p + 1;
        link_crit = linkage_crit(x[0], x[1]);
        if p == 1:
            point = link_crit;
            linkage_new = list(itertools.chain(x[0], x[1]));
            r = x[0];
            c = x[1];
        elif point > link_crit:
            point = link_crit;
            linkage_new = list(itertools.chain(x[0], x[1]));
            r = x[0];
            c = x[1];
    print('STEP :', steps)
    steps = steps + 1
    print ('Clusters merged : ', linkage_new)
    print ('with Linkage criterion : ', point, '\n')
    clusters.remove(r);
    clusters.remove(c);
    clusters.append(linkage_new);
    print ('Current cluster: ', clusters, "\n")  
    