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
print(np.shape(df))

print(df)


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
y = df["lameLeg"]
ydict = {"none": 1, "left:hind": 2, "left:fore": 3, "right:hind": 4, "right:fore": 5}
yreal = np.empty((len(y), 1))
for i, e in enumerate(y):
    yreal[i] = int(ydict[e])
#print(yreal)

kf = KFold(n_splits=5, shuffle = True)

# https://stackoverflow.com/questions/51852551/key-error-not-in-index-while-cross-validation

acc_AW = []

for i, (train_index, test_index) in enumerate(kf.split(X_AW, yreal)):
    X_train = X_AW[train_index,:]
    y_train = yreal[train_index]
    
    X_test = X_AW[test_index,:]
    y_test = yreal[test_index]
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    test_preds = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, test_preds)
    
    acc_AW.append(accuracy)
    

print("############################")

acc_PC34 = []

for i, (train_index, test_index) in enumerate(kf.split(X_AW, yreal)):
    X_train = X_PC34[train_index,:]
    y_train = yreal[train_index]
    
    X_test = X_PC34[test_index,:]
    y_test = yreal[test_index]
    
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    
    test_preds = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, test_preds)

    acc_PC34.append(accuracy)
    
    
print(np.mean(acc_AW))
print(np.mean(acc_PC34))
    
    
    
    
    
    
    
    