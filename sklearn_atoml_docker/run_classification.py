import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
# own imports
import sklearn_classifiers

# set variables
rndSeed = 42
datasetName = 'wine'
targetName = 'quality'

# set data folder
dataPath = 'data/'+datasetName+'/'

# read the split dataset
df_train = pd.read_csv('/data/wine_train_sklearn.csv')
df_test = pd.read_csv('/data/wine_test_sklearn.csv')

# split target from features
X_train = df_train.drop(targetName, axis=1)
t_train = df_train[targetName]
X_test = df_test.drop(targetName, axis=1)
t_test = df_test[targetName]

# run classifier function
#y_pred_sklearn = sklearn_classifiers.sklearn_GaussianNB(X_train, t_train, X_test)
#y_pred_sklearn = sklearn_classifiers.sklearn_MultinomialNB(X_train, t_train, X_test)
y_pred_sklearn = sklearn_classifiers.sklearn_LogisticRegression(X_train, t_train, X_test)
#y_pred_sklearn = sklearn_classifiers.sklearn_LinearSVM(X_train, t_train, X_test)

# print classification report
#print(classification_report(t_test, y_pred_sklearn, digits=8))
report = classification_report(t_test, y_pred_sklearn, digits=6, output_dict=True)
print('accuracy: %f    f1: %f\n' % (report['accuracy'], report['weighted avg']['f1-score']))
# print confusion matrix
print('Confusion matrix: true/prediction:')
print(confusion_matrix(t_test, y_pred_sklearn))

# save predictions
# originalpath: dataPath + datasetName + '_sklearn_pred.csv'
#path = "log/"+ datasetName + '_sklearn_pred.csv'
df_pred = pd.DataFrame(y_pred_sklearn)
df_pred.to_csv('/log/sklearn_pred.csv', header=True, index=False)
#print(path)
print(os.getcwd())
