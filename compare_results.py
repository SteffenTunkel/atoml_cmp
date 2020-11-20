import os
import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('algorithm-descriptions/framework_matching.csv')
print(df)


for index, row in df.iterrows():
	algorith_ident = row['sklearn']

	print('\nThe predictions in the following files are compared with each other:')

	sklearn_csvfile = 'predictions/sklearn/pred_SKLEARN_' + row['sklearn'] + '_Uniform.csv'
	print(sklearn_csvfile)
	pred_sklearn = pd.read_csv(sklearn_csvfile, header=None)

	weka_csvfile = 'predictions/weka/pred_WEKA_' + row['weka'] + '_Uniform.csv'
	print(weka_csvfile)
	pred_weka = pd.read_csv(weka_csvfile)
	pred_weka = pred_weka["prediction"]
	
	spark_path= 'predictions/spark/pred_SPARK_' + row['spark'] + '_Uniform'
	spark_csv_folder = [f for f in os.listdir(spark_path) if os.path.isfile(os.path.join(spark_path, f))]
	#print(spark_csv_folder)
	for file in spark_csv_folder:
		if file.endswith(".csv"):
			print(spark_path + '/' + file)
			pred_spark = pd.read_csv(spark_path + '/' + file)
	print('\n')


	# read the prediction csvs manually
	#pred_sklearn = pd.read_csv('predictions/sklearn/pred_SKLEARN_LogisticRegression_Uniform.csv', header=None)
	#pred_spark = pd.read_csv('predictions/spark/pred_SPARK_LogisticRegression_Uniform/part-00000-b265813f-c360-4e21-9269-610b95f4bc07-c000.csv')
	#pred_weka = pd.read_csv('predictions/weka/pred_WEKA_Logistic_Uniform.csv')
	
	# Take a look at the files
	#print(pred_sklearn.head(6))
	#print(pred_spark.head(6))
	#print(pred_weka.head(6))

	# Label Encoder, if needed
	#le = LabelEncoder()
	#pred_sklearn = le.fit_transform(pred_sklearn)


	# calculate and print the confusion matix
	print('Comparation between SKLEARN and SPARK:', index, algorith_ident)
	print(confusion_matrix(pred_sklearn, pred_spark))

	# calculate and print the confusion matix
	print('Comparation between SKLEARN and WEKA:', index, algorith_ident)
	print(confusion_matrix(pred_sklearn, pred_weka))
