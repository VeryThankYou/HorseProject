import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut

df = pd.read_csv("horse_data23.txt", sep = "\t")

dfone=df
horses=[1,2,3,4,5,6,7,9]
index_list=list(df.values.tolist())
for i in range(len(horses)):
    horse="B"+str(horses[i])
    for k in range(len(df)):
        tempList=list()
        if(index_list[k][1]==(horse)):
            print(index_list[k])
        else:
            tempList.append(index_list[k])
    print(tempList)        
        #print(dfone)
    