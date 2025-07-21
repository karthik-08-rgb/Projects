import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

heartdata = pd.read_csv("heart.csv")

X = heartdata.drop(columns='target', axis=1)
y = heartdata['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree accuracy:", acc_dt)

from sklearn.metrics import f1_score, confusion_matrix

f1_dt = f1_score(y_test, y_pred_dt)
print("Decision Tree F1 score:", f1_dt)

cm_dt = confusion_matrix(y_test, y_pred_dt)
print("Decision Tree confusion matrix:")
print(cm_dt)

from sklearn.tree import export_graphviz
from graphviz import Source
import matplotlib.pyplot as plt

dot_data = export_graphviz(dt, out_file=None, feature_names=X.columns, class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
graph = Source(dot_data)

plt.figure(figsize=(15,10))
plt.axis('off')
plt.imshow(graph.render(format='png'))
plt.show()
