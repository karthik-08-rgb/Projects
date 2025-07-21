import numpy as np
import pandas as py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

heartdata=py.read_csv("heart.csv")

heartdata.head()
heartdata.tail()
# heartdata.shape



heartdata.info()
heartdata.describe()
targets=heartdata['target'].value_counts()

#all columns
X=heartdata.drop(columns='target',axis=1)
#target column
Y=heartdata['target']


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
# print(X.shape,X_train.shape,X_test.shape)





model=LogisticRegression()
model.fit(X_train,Y_train )


X_train_prediction=model.predict(X_train)
trainigdataaccuracy=accuracy_score(X_train_prediction,Y_train)
# print( trainigdataaccuracy)




X_test_prediction=model.predict(X_test)
testdataaccuracy=accuracy_score(X_test_prediction,Y_test)
print( testdataaccuracy)



input_from_user=(71,0,0,112,149,0,1,125,0,1.6,1,0,2)
input_from_user_array=np.asarray(input_from_user)
input_from_user_reshaped=input_from_user_array.reshape(1,-1)
prediction=model.predict(input_from_user_reshaped)


if prediction[0]==0:
    print("Patient Doesnot have  Any Heart Dieseas")
else:
    print("Patient Has heart dieseas he needs more tests")    




Y_test_probabilities = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, Y_test_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()




# train_accuracies = []
# test_accuracies = []
# for i in range(1, 21):
#     model.set_params(max_iter=i)
#     model.fit(X_train, Y_train)
#     Y_train_predictions = model.predict(X_train)
#     Y_test_predictions = model.predict(X_test)
#     train_accuracy = accuracy_score(Y_train_predictions, Y_train)
#     test_accuracy = accuracy_score(Y_test_predictions, Y_test)
#     train_accuracies.append(train_accuracy)
#     test_accuracies.append(test_accuracy)

# plt.plot(range(1, 21), train_accuracies, label='Training Accuracy')
# plt.plot(range(1, 21), test_accuracies, label='Testing Accuracy')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Accuracy')
# plt.title('Training Progress of Logistic Regression Model')
# plt.legend(loc='lower right')
# plt.show()




# sns.countplot(data= heartdata, x='target', palette="mako")
# plt.xlabel('Has Heart Disease (1 = Yes, 0 = No)')
# plt.ylabel('Count')
# plt.title('Distribution of Heart Disease in Dataset')
# plt.show()



# Plot the heatmap
# fig, ax = plt.subplots(figsize=(15, 15))
# sns.heatmap(heartdata.corr(),ax=ax, annot=True)
# # Show the plot
# plt.show()
