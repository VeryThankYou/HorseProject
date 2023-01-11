#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:36:26 2023

@author: clarasofiechristiansen
"""

import pandas as pd
import numpy as np

df = pd.read_csv("predictions_section2_horseout.csv")


d = {}

names = ["True", "Baseline", "AW", "PC3PC4", "Both"]

for i in range(1,len(names)):
    for j in range(i+1, len(names)):
        #n11
        n11 = sum((df[names[i]] == df["True"]) * (df[names[j]] == df["True"]))

        #n12 # Baseline True
        n12 = sum((df[names[i]] == df["True"]) * (df[names[j]] != df["True"]))

        #n21 # A/W True
        n21 = sum((df[names[i]] != df["True"]) * (df[names[j]] == df["True"]))

        #n22
        n22 = sum((df[names[i]] != df["True"]) * (df[names[j]] != df["True"]))
        
        print([[n11,n12],[n21,n22]])

        matrix = pd.DataFrame([[n11,n12],[n21,n22]])
        matrix = matrix.rename(columns={0: "int1", 1: "int2"}, index={0: "int1", 1: "int2"})
        d[names[i] + "-" + names[j]] = matrix
        
print(d)

for i in d:
    d[i].to_csv(i + ".csv")
