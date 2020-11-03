# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 06:33:29 2020

@author: Pyare
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve

data = pd.read_csv('data_7.1.csv');

              
X = data.loc[:,'A':'weights'];
y = data.loc[:,'class'];

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)
print(X_train.shape)

#remove zero variance
sel_variance_threshold = VarianceThreshold() 
X_train_remove_variance = sel_variance_threshold.fit_transform(X_train)
print(X_train_remove_variance.shape)

#Tree based
# model_tree = RandomForestClassifier(random_state=100, n_estimators= 50)
# sel_rfe_tree = RFE(estimator=model_tree, n_features_to_select= 20, step=1)
# X_train_rfe_tree = sel_rfe_tree.fit_transform(X_train, y_train)
# print(sel_rfe_tree.get_support())
# randF = sel_rfe_tree.get_support()

#chi2
sel_chi2 = SelectKBest(chi2, k=20)   
X_train_chi2 = sel_chi2.fit_transform(X_train, y_train)
print(sel_chi2.get_support())
chi2_sup = sel_chi2.get_support()

#MI
sel_mutual = SelectKBest(mutual_info_classif, k= 20)
X_train_mutual = sel_mutual.fit_transform(X_train, y_train)
print(sel_mutual.get_support())
mi_sup = sel_mutual.get_support()

rfc = RandomForestClassifier(n_estimators=50,
                             criterion='gini',
                             max_depth=5,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0,
                             max_features='auto',
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_impurity_split=None,
                             bootstrap=True,
                             oob_score=False,
                             n_jobs=-1,
                             verbose=0,
                             warm_start=False,
                             class_weight='balanced');

rfc.fit(X_train_mutual, y_train);
X_test_mutual = sel_mutual.transform(X_test)
print(X_test.shape)
print(X_test_mutual.shape)

y_pred = rfc.predict(X_test_mutual);

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_proba = rfc.predict_proba(X_test_mutual)[:,1];

#MSE
mse = mean_squared_error(y_test, y_pred);
print('MSE = ' + str(mse*100) +'%');

#Confusion Matrix
cf = confusion_matrix(y_test, y_pred, labels=[0, 1]);
tn, fp, fn, tp = cf.ravel();
(tn, fp, fn, tp)

#Accuracy
accuracy = accuracy_score(y_test, y_pred);
print('Accuracy = ' + str(accuracy));

#Senstivity
sensitivity = tp/(tp + fn);
print('Sensitivity = ' + str(sensitivity));

#Specificity
specificity = tn/(tn + fp);
print('Specificity = ' + str(specificity));

#Precision
precision = precision_score(y_test, y_pred, average='micro');
print('Precision = ' + str(precision));

#Recall
recall = recall_score(y_test, y_pred, average='micro');
print('Recall = ' + str(recall));

#PR Curve
average_precision = average_precision_score(y_test, y_proba);
disp2 = plot_precision_recall_curve(rfc, X_test_mutual, y_test)
disp2.ax_.set_title('Binary Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


#AUC-ROC
roc_value = roc_auc_score(y_test, y_proba);

#ROC Curve
rfc_disp = plot_roc_curve(rfc, X_test_mutual, y_test, alpha = 0.8);
plt.show()

#F1-Score
f1 = f1_score(y_test, y_pred, average='micro');
print('F1-score = ' + str(f1));

#MCC - Matthews correlation coefficient 
mcc = matthews_corrcoef(y_test, y_pred);
print('MCC = ' + str(mcc));

#kappa statistic
ck = cohen_kappa_score(y_test, y_pred);
print('Kappa = ' + str(ck));