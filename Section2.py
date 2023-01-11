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
accuracy_outerBL = np.zeros((1,k1))

all_test_predsBL = np.array([])
all_test_predsAW = np.array([])
all_test_predsPC34 = np.array([])
all_test_predsBoth = np.array([])
all_test_true = np.array([])


for i, (par_index, test_index) in enumerate(kf1.split(X_AW, yreal)):
    print(par_index)
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
    all_test_predsAW = np.append(all_test_predsAW, test_predsAW, axis = 0)
    
    accuracy = accuracy_score(y_test, test_predsAW)
    accuracy_outerAW[0,i] = best_k
    accuracy_outerAW[1,i] = accuracy
    
    
    # PC34
    mean_accuracy = np.mean(accuracy_innerPC34, axis = 0)
    best_k = complexity[np.argmax(mean_accuracy)]
    
    knn_modelPC34 = KNeighborsClassifier(n_neighbors=best_k)
    knn_modelPC34.fit(X_parPC34, y_par)
    test_predsPC34 = knn_modelPC34.predict(X_testPC34)
    all_test_predsPC34 = np.append(all_test_predsPC34, test_predsPC34, axis = 0)
    
    accuracy = accuracy_score(y_test, test_predsPC34)
    accuracy_outerPC34[0,i] = best_k
    accuracy_outerPC34[1,i] = accuracy
    
    
    # Both
    mean_accuracy = np.mean(accuracy_innerBoth, axis = 0)
    best_k = complexity[np.argmax(mean_accuracy)]
    
    knn_modelBoth = KNeighborsClassifier(n_neighbors=best_k)
    knn_modelBoth.fit(X_parBoth, y_par)
    test_predsBoth = knn_modelBoth.predict(X_testBoth)
    all_test_predsBoth = np.append(all_test_predsBoth, test_predsBoth, axis = 0)
    
    accuracy = accuracy_score(y_test, test_predsBoth)
    accuracy_outerBoth[0,i] = best_k
    accuracy_outerBoth[1,i] = accuracy

    # Baseline
    choice = np.bincount(y_par.astype(int)).argmax()
    BL_pred = np.full(len(y_test), choice)
    all_test_predsBL = np.append(all_test_predsBL, BL_pred, axis = 0)
    
    BL_accuracy = accuracy_score(y_test.astype(int), BL_pred)
    accuracy_outerBL[0, i] = BL_accuracy
    
    # True class
    all_test_true = np.append(all_test_true, y_test, axis = 0)
            
            

# Predictions
d = {"True": all_test_true, "Baseline": all_test_predsBL, "AW": all_test_predsAW,"PC3PC4": all_test_predsPC34, "Both": all_test_predsBoth}
df = pd.DataFrame(d)
print(df)
df.to_csv("predictions_section2.csv")

# Accuracy

print(accuracy_outerAW)
print(accuracy_outerPC34)
print(accuracy_outerBoth)
print(accuracy_outerBL)

mean_acc_AW = np.mean(accuracy_outerAW[1,:])
mean_acc_PC34 = np.mean(accuracy_outerPC34[1,:])
mean_acc_Both = np.mean(accuracy_outerBoth[1,:])
mean_acc_BL = np.mean(accuracy_outerBL)

# For McNemar
print(mean_acc_AW)
print(mean_acc_PC34)
print(mean_acc_Both)
print(mean_acc_BL)

print()
d = {"Baseline": [mean_acc_BL], "A/W": [mean_acc_AW], "PC3/PC4": [mean_acc_PC34], "A/W/PC3/PC4": [mean_acc_Both]}
df = pd.DataFrame(d)
print(df)

df.to_csv("mean_acc.csv")
