import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the heart disease dataset
heartdata = pd.read_csv("heart.csv")

# Separate features and target
X = heartdata.drop(columns='target', axis=1)
y = heartdata['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Create and fit a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_rf = rf_model.predict(X_test)

# Calculate the accuracy of the model
acc_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest accuracy:", acc_rf)

# Make predictions on a single example
example = [[71, 0, 0, 112, 149, 0, 1, 125, 0, 1.6, 1, 0, 2]]
prediction = rf_model.predict(example)
if prediction[0] == 0:
    print("Patient does not have any heart disease")
else:
    print("Patient has heart disease and needs more tests")

# Calculate the f1 score
f1_rf = classification_report(y_test, y_pred_rf)
print("Random Forest f1 score:")
print(f1_rf)

# Calculate the confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest confusion matrix:")
print(cm_rf)
