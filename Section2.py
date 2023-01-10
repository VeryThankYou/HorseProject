#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 07:57:44 2023

@author: clarasofiechristiansen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

df = pd.read_csv("horse_data23.txt", sep = "\t")


# Prepare label/color
label_to_color = {'none': 'blue', 'right:hind': 'red', 'right:fore': 'green', 'left:hind': 'yellow', 'left:fore': 'purple'}
colors = [label_to_color[label] for label in df["lameLeg"]]



# Scatter plot A/W and lameness
plt.scatter(df["A"], df["W"], c = colors)

# Extra
plt.title('Scatter Plot')
plt.xlabel('A')
plt.ylabel('W')
legend_elements = [plt.scatter([], [], c=color, label=label) for label, color in label_to_color.items()]
plt.legend(handles=legend_elements)
plt.show()


# Scatter plot PC3/PC4 and lameness
# Prepare A/W data
plt.scatter(df["pc3"],df["pc4"], c = colors)

# Extra
plt.title('Scatter Plot')
plt.xlabel('PC3')
plt.ylabel('PC4')
legend_elements = [plt.scatter([], [], c=color, label=label) for label, color in label_to_color.items()]
plt.legend(handles=legend_elements)
plt.show()


# Prepare data for KNN (AW)
X_AW = np.transpose(np.array([df["A"], df["W"]]))
X_PC34 = np.transpose(np.array([df["pc3"], df["pc4"]]))
X_Both = np.transpose(np.array([df["A"], df["W"], df["pc3"], df["pc4"]]))
y = df["lameLeg"]
ydict = {"none": 1, "left:hind": 2, "left:fore": 3, "right:hind": 4, "right:fore": 5}
yreal = np.empty((len(y)))
for i, e in enumerate(y):
    yreal[i] = int(ydict[e])
#print(yreal)

k1 = 10
k2 = 5

kf1 = KFold(n_splits=k1, shuffle = True)
kf2 = KFold(n_splits=k2, shuffle = True)
complexity = [1, 3, 5, 7, 9]

# https://stackoverflow.com/questions/51852551/key-error-not-in-index-while-cross-validation


accuracy_outerAW = np.zeros((2,k1))
accuracy_outerPC34 = np.zeros((2,k1))
accuracy_outerBoth = np.zeros((2,k1))

for i, (par_index, test_index) in enumerate(kf1.split(X_AW, yreal)):
    
    y_par = yreal[par_index]
    y_test = yreal[test_index]
    
    X_parAW = X_AW[par_index,:]
    X_testAW = X_AW[test_index,:]
    
    X_parPC34 = X_PC34[par_index,:]
    X_testPC34 = X_PC34[test_index,:]
    
    X_parBoth = X_Both[par_index,:]
    X_testBoth = X_Both[test_index,:]
    
    accuracy_innerAW = np.zeros((k2,len(complexity)))
    accuracy_innerPC34 = np.zeros((k2,len(complexity)))
    accuracy_innerBoth = np.zeros((k2,len(complexity)))
    
    for j, (train_index, val_index) in enumerate(kf2.split(X_parAW, y_par)):
        y_train = y_par[train_index]
        y_val = y_par[val_index]
        
        X_trainAW = X_parAW[train_index,:]
        X_valAW = X_parAW[val_index,:]
        
        X_trainPC34 = X_parPC34[train_index,:]
        X_valPC34 = X_parPC34[val_index,:]
        
        X_trainBoth = X_parBoth[train_index,:]
        X_valBoth = X_parBoth[val_index,:]
        
        
        for k in range(len(complexity)):
            knn_modelAW = KNeighborsClassifier(n_neighbors=complexity[k])
            knn_modelAW.fit(X_trainAW, y_train)
            test_predsAW = knn_modelAW.predict(X_valAW)
            accuracy = accuracy_score(y_val, test_predsAW)
            accuracy_innerAW[j,k] = accuracy
            
            
            knn_modelPC34 = KNeighborsClassifier(n_neighbors=complexity[k])
            knn_modelPC34.fit(X_trainPC34, y_train)
            test_predsPC34 = knn_modelPC34.predict(X_valPC34)
            accuracy = accuracy_score(y_val, test_predsPC34)
            accuracy_innerPC34[j,k] = accuracy
            
            
            knn_modelBoth = KNeighborsClassifier(n_neighbors=complexity[k])
            knn_modelBoth.fit(X_trainBoth, y_train)
            test_predsBoth = knn_modelBoth.predict(X_valBoth)
            accuracy = accuracy_score(y_val, test_predsBoth)
            accuracy_innerBoth[j,k] = accuracy
        
    # AW
    mean_accuracy = np.mean(accuracy_innerAW, axis = 0)
    best_k = complexity[np.argmax(mean_accuracy)]
    
    knn_modelAW = KNeighborsClassifier(n_neighbors=best_k)
    knn_modelAW.fit(X_parAW, y_par)
    test_predsAW = knn_modelAW.predict(X_testAW)
    
    accuracy = accuracy_score(y_test, test_predsAW)
    accuracy_outerAW[0,i] = best_k
    accuracy_outerAW[1,i] = accuracy
    
    
    # PC34
    mean_accuracy = np.mean(accuracy_innerPC34, axis = 0)
    best_k = complexity[np.argmax(mean_accuracy)]
    
    knn_modelPC34 = KNeighborsClassifier(n_neighbors=best_k)
    knn_modelPC34.fit(X_parPC34, y_par)
    test_predsPC34 = knn_modelPC34.predict(X_testPC34)
    accuracy = accuracy_score(y_test, test_predsPC34)
    accuracy_outerPC34[0,i] = best_k
    accuracy_outerPC34[1,i] = accuracy
    
    
    # Both
    mean_accuracy = np.mean(accuracy_innerBoth, axis = 0)
    best_k = complexity[np.argmax(mean_accuracy)]
    
    knn_modelBoth = KNeighborsClassifier(n_neighbors=best_k)
    knn_modelBoth.fit(X_parBoth, y_par)
    test_predsBoth = knn_modelBoth.predict(X_testBoth)
    accuracy = accuracy_score(y_test, test_predsBoth)
    accuracy_outerBoth[0,i] = best_k
    accuracy_outerBoth[1,i] = accuracy
            
            
            
    
print(accuracy_outerAW)
print(accuracy_outerPC34)
print(accuracy_outerBoth)
    
    