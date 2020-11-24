# at the moment limited to the Uniform data

import os
import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import LabelEncoder


def createConfusionMatrix(algorithm_identifier, predictions, frameworks, df, iteration):
	print('%d: Comparation between %s and %s for %s' % (iteration, frameworks[0].upper(), frameworks[1].upper(), algorithm_identifier))
	cm = confusion_matrix(predictions[0], predictions[1])
	print(cm)
	if cm[0,1] == 0 and cm[1,0] == 0 and ( cm[0,0] != 0 or cm[1,1] != 0):
		same = True
	else:
		same = False
	frame_data = [[same, cm[0,0], cm[0,1], cm[1,0], cm[1,1], ('%s_%s_%s'%(frameworks[0],frameworks[1],algorithm_identifier))]]
	df = df.append(pd.DataFrame(data=frame_data, columns=df.columns), ignore_index=True)
	return df, same, cm


print('MatchingTable:')

matching_table = pd.read_csv('algorithm-descriptions/framework_matching.csv')
print(matching_table)

result_columns = ["same","TP","FP","TN","FN","compared_algorithm"]
results_df = pd.DataFrame(columns=result_columns)
print(results_df)
for index, row in matching_table.iterrows():
	algorithm_ident = row['sklearn']

	print('\nThe predictions in the following files are compared with each other:')

	# get sklearn csv file
	sklearn_csvfile = 'predictions/sklearn/pred_SKLEARN_' + row['sklearn'] + '_Uniform.csv'
	print(sklearn_csvfile)
	pred_sklearn = pd.read_csv(sklearn_csvfile, header=None)

	# get weka csv file
	weka_csvfile = 'predictions/weka/pred_WEKA_' + row['weka'] + '_Uniform.csv'
	print(weka_csvfile)
	pred_weka = pd.read_csv(weka_csvfile)
	pred_weka = pred_weka["prediction"]
	
	# spark CSVs are currently saved in a weird format (csv file is in a named folder)
	spark_path= 'predictions/spark/pred_SPARK_' + row['spark'] + '_Uniform'
	spark_csv_folder = [f for f in os.listdir(spark_path) if os.path.isfile(os.path.join(spark_path, f))]
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

	results_df, _, _=createConfusionMatrix(algorithm_ident, [pred_sklearn, pred_spark], ["sklearn", "spark"], results_df, index)
	results_df, _, _=createConfusionMatrix(algorithm_ident, [pred_sklearn, pred_weka], ["sklearn", "weka"], results_df, index)
	results_df, _, _=createConfusionMatrix(algorithm_ident, [pred_spark, pred_weka], ["spark", "weka"], results_df, index)

print("\n ResultDataFrame:")
print(results_df)
results_df.to_csv('algorithm-descriptions/framework_matching_results.csv', index=False)


