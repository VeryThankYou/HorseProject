import numpy as np
import pandas as pd

df = pd.read_csv("predictions_section2_horseout_collapse.csv")
print(df)

print("---- Baseline ----")

numType2ErrorBL = np.sum((df["True"] != df["Baseline"]) * (df["Baseline"] == 1))
numErrorBL = np.sum((df["True"] != df["Baseline"]))

print("Percentage of type 2 errors in full prediction")
print(numType2ErrorBL/85)

print("Percentage of type 2 errors in errors")
print(numType2ErrorBL/numErrorBL)

print("---- AW model ----")

numType2ErrorAW = np.sum((df["True"] != df["AW"]) * (df["AW"] == 1))
numErrorAW = np.sum((df["True"] != df["AW"]))

print("Percentage of type 2 errors in full prediction")
print(numType2ErrorAW/85)

print("Percentage of type 2 errors in errors")
print(numType2ErrorAW/numErrorAW)

print("---- PC3PC4 model ----")
numType2ErrorPC = np.sum((df["True"] != df["PC3PC4"]) * (df["PC3PC4"] == 1))
numErrorPC = np.sum((df["True"] != df["PC3PC4"]))

print("Percentage of type 2 errors in full prediction")
print(numType2ErrorPC/85)
print("Percentage of type 2 errors in errors")
print(numType2ErrorPC/numErrorPC)

print("---- Both model ----")
numType2ErrorBoth = np.sum((df["True"] != df["Both"]) * (df["Both"] == 1))
numErrorBoth = np.sum((df["True"] != df["Both"]))

print("Percentage of type 2 errors in full prediction")
print(numType2ErrorBoth/85)
print("Percentage of type 2 errors in errors")
print(numType2ErrorBoth/numErrorBoth)