import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

df = pd.read_csv("horse_data23.txt", sep = "\t")
horses=[1,2,3,4,5,6,7,9]
index_list=list(df.values.tolist())
for i in range(len(horses)):
    horse="B"+str(horses[i])
    templist=list()
    for k in range(len(df)):
        if(index_list[k][1]!=(horse)):
            templist.append(index_list[k])
    print(templist)
    print(i)

    
    
    