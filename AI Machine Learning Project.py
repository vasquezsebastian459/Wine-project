from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import os

sns.set_theme()
IMAGES = "images"
if not os.path.exists("images"):
    os.mkdir(IMAGES)
df = pd.read_csv("winequality-red.csv")

print("describe")
df.describe()
print("info")
df.info()

fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
  if col != "type":
    sns.histplot(value, ax=ax[index])
    index += 1
plt.tight_layout(pad = 0.5, w_pad=0.7, h_pad=5.0)
plt.savefig(f"{IMAGES}/histograms.png")
plt.show()


df.head()

df.isnull().sum()

lin = sns.scatterplot(data = df, x= "quality", y= "total sulfur dioxide")

acidityplot = plt.figure(figsize=(5,5))
sns.barplot(x="quality", palette="rocket",y= "volatile acidity",data = df)

plt.savefig(f"{IMAGES}/quality_vol_acid bar plot.png")
plt.show()

citricacid = plt.figure(figsize=(5,5))
sns.barplot(x="quality", palette="rocket",y= "citric acid",data = df)
plt.savefig(f"{IMAGES}/quality_citric_acid bar plot.png")
plt.show()


#sns.histplot(df,x= quality)
#plt.savefig(f"{IMAGES}/quality_histogram.png")
#plt.show()


correlation = df.corr()

#Creating a heatmap to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, cmap="BuPu",square=True, fmt='.2f', annot=True, annot_kws={'size':8})
plt.savefig(f"{IMAGES}/correlation_matrix.png")
plt.show()

#DATA PREPROCESSING
#separate the data and label
#store quality differently to later feed our  Model
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
X = df.drop(['quality', 'best quality'],axis=1)
Y = df['best quality']

print("X:",X)
print("Y:", Y)

#TRAIN & TEST SPLIT
print("TRAIN & TEST SPLIT")
#train ML
#evaluate with test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
#20% of the data to be tested / 80% training.

print("Y.shape", "Y_train.shape", "Y_test.shape")
print(Y.shape, Y_train.shape, Y_test.shape)

#Normalising data

print("Normalising data")
norm = MinMaxScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)


print(X_train)
print(X_test)



#MODEL TRAINING


#Random Forest Classifier
#ensemble model of different decision trees
model = RandomForestClassifier()
model.fit(X_train,Y_train)


print("MODEL: RANDOM FORREST")
#MODEL EVALUATION random forest
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print(classification_report(Y_test, X_test_prediction))
print("Accuracy forest:", test_data_accuracy)

rfc_eval = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("forest rfc_eval mean:", rfc_eval.mean())


metrics.plot_confusion_matrix(model, X_test, Y_test, cmap='BuPu')
plt.savefig(f"{IMAGES}/confusion_matrix_random_forest.png")
plt.show()

#define metrics
y_pred_proba = model.predict_proba(X_test)[::,1]
print("forest y_pred_proba", y_pred_proba)
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('')
plt.xlabel('')
plt.savefig(f"{IMAGES}/forest_ROC_curve.png")
plt.show()

#calculate are under the curve (AUC)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
print("auc", auc)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig(f"{IMAGES}/forest_roc_auc_curve.png")
plt.show()

# RESULTS model forest
X_test_prediction = list(model.predict(X_test))
print("X_test_prediction")
print(X_test_prediction)
predicted_df_forest = {'predicted_values': X_test_prediction, 'original_values': Y_test}
predicted_df_forest = pd.DataFrame(predicted_df_forest)
print("Forest predicted df")
print(predicted_df_forest.head(20))


#SGDClassifier
print("MODEL: SGDClassifier")
model = SGDClassifier(loss="modified_huber")
model.fit(X_train,Y_train)

#MODEL EVALUATION sgd
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print(classification_report(Y_test, X_test_prediction))
print("Accuracy SGD:", test_data_accuracy)

rfc_eval = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("SGD rfc_eval mean:", rfc_eval.mean())

metrics.plot_confusion_matrix(model, X_test, Y_test, cmap='BuPu')
plt.savefig(f"{IMAGES}/confusion_matrix_SGD.png")
plt.show()

#define metrics

y_pred_proba = model.predict_proba(X_test)[::,1]
print("SGD y_pred_proba", y_pred_proba)
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('')
plt.xlabel('')
plt.savefig(f"{IMAGES}/SGD_ROC_curve.png")
plt.show()

#calculate are under the curve (AUC)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
print("auc", auc)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig(f"{IMAGES}/SGD_roc_auc_curve.png")
plt.show()

# RESULTS model SGD
X_test_prediction = list(model.predict(X_test))
print("X_test_prediction")
print(X_test_prediction)
predicted_df_SGD = {'predicted_values': X_test_prediction, 'original_values': Y_test}
predicted_df_SGD = pd.DataFrame(predicted_df_SGD)
print("Forest predicted SGD")
print(predicted_df_SGD.head(20))

#LogisticRegression
print("MODEL: LogisticRegression")
model = LogisticRegression()
model.fit(X_train,Y_train)

#MODEL EVALUATION logistic
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print(classification_report(Y_test, X_test_prediction))
print("Accuracy logistic:", test_data_accuracy)

rfc_eval = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10)
print("log rfc_eval mean:", rfc_eval.mean())

metrics.plot_confusion_matrix(model, X_test, Y_test, cmap='BuPu')
plt.savefig(f"{IMAGES}/confusion_matrix_logistic_regression.png")
plt.show()

#define metrics

y_pred_proba = model.predict_proba(X_test)[::,1]
print("log y_pred_proba", y_pred_proba)
fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
#create ROC curve
plt.plot(fpr,tpr)
plt.ylabel('')
plt.xlabel('')
plt.savefig(f"{IMAGES}/Log_ROC_curve.png")
plt.show()

#calculate are under the curve (AUC)
auc = metrics.roc_auc_score(Y_test, y_pred_proba)
print("auc", auc)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig(f"{IMAGES}/log_roc_auc_curve.png")
plt.show()

# RESULTS model
X_test_prediction = list(model.predict(X_test))
print("X_test_prediction")
print(X_test_prediction)
predicted_df_log = {'predicted_values': X_test_prediction, 'original_values': Y_test}
predicted_df_log = pd.DataFrame(predicted_df_log)
print("Log predicted df")
print(predicted_df_log.head(20))