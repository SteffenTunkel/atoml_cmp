import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# read the prediction csvs
pred_sklearn = pd.read_csv('predictions/sklearn_pred.csv')
pred_spark = pd.read_csv('predictions/spark_pred_logreg.csv')
pred_weka = pd.read_csv('predictions/weka_pred.csv')

# preprocess the predictions
le = LabelEncoder()
pred_sklearn = le.fit_transform(pred_sklearn)
#pred_spark = pred_spark['prediction']


# calculate and print the confusion matix
print('Compare Sklearn and Spark framework predictions:')
print(confusion_matrix(pred_sklearn, pred_spark['prediction']))

# calculate and print the confusion matix
print('Compare Sklearn and Weka framework predictions:')
print(confusion_matrix(pred_sklearn, pred_spark['prediction']))
