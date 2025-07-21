import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the heart disease dataset
heartdata = pd.read_csv("heart.csv")

# Separate features and target
X = heartdata.drop(columns='target', axis=1)
y = heartdata['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Create and fit a KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_knn = knn_model.predict(X_test)

# Calculate the accuracy of the model
acc_knn = accuracy_score(y_test, y_pred_knn)
print("KNN accuracy:", acc_knn)

# Make predictions on a single example
example = [[71, 0, 0, 112, 149, 0, 1, 125, 0, 1.6, 1, 0, 2]]
prediction = knn_model.predict(example)
if prediction[0] == 0:
    print("Patient does not have any heart disease")
else:
    print("Patient has heart disease and needs more tests")

# Calculate the f1 score
f1_knn = classification_report(y_test, y_pred_knn)
print("KNN f1 score:")
print(f1_knn)

# Calculate the confusion matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("KNN confusion matrix:")
print(cm_knn)
