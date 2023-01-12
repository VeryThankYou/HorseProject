import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

df = pd.read_csv("horse_data23.txt", sep = "\t")

# Plot decision boundaries
# Prepare label/color
label_to_color = {'none': '#322BFF', "left:hind": '#FF1125', "left:fore": '#1ACE1D', "right:hind": '#FFFF00', "right:fore": '#8954CE'}


colors = [label_to_color[label] for label in df["lameLeg"]]

X_AW = np.transpose(np.array([df["A"], df["W"]]))
y = df["lameLeg"]
ydict = {"none": 1, "left:hind": 2, "left:fore": 3, "right:hind": 4, "right:fore": 5}
yreal = np.empty((len(y)))
for i, e in enumerate(y):
    yreal[i] = int(ydict[e])

xx, yy = np.meshgrid(np.arange(-2, 1, 0.005),np.arange(-0.25, 0.25, 0.001))


model = KNeighborsClassifier(5)
model.fit(X_AW, yreal)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
print(Z)
cmap_light = ListedColormap(['#7F7CFF', '#FF9EA7', '#86CC87', "#FDFFA8", "#AE96CE"])
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(df["A"], df["W"], c = colors)
# Extra
plt.title('Scatter Plot')
plt.xlabel('A')
plt.ylabel('W')
legend_elements = [plt.scatter([], [], c=color, label=label) for label, color in label_to_color.items()]
plt.legend(handles=legend_elements)
plt.show()
    