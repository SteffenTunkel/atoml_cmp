# at the moment limited to the Uniform data

import os
import pandas as pd
import numpy as np
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


def jsonToCsvForSpark():
	# Function for the conversion of the json file created by the atoml spark docker
	#	to csv files, which are similar to the ones of sklearn.

	sparkPath = 'predictions/spark/'
	sparkJsonPath = sparkPath + "json/"

	# get all json folder names
	sparkJsonList = [f for f in os.listdir(sparkJsonPath)]
	print("Convert json spark files into csv. Files found:")
	print(sparkJsonList)
	for jsonFolder in sparkJsonList:
		oneJsonFolderContent = [f for f in os.listdir(sparkJsonPath + jsonFolder) if os.path.isfile(os.path.join(sparkJsonPath + jsonFolder, f))]
		
		# load a json file, process it and save it as a csv
		for file in oneJsonFolderContent:
			if file.endswith(".json"):
				pred_spark = pd.read_json((sparkJsonPath + jsonFolder + "/" + file), lines=True)
				prob_0 = []
				prob_1 = []
				for dictionary in pred_spark.probability.tolist():
					prob_0.append(dictionary.get("values")[0])
					prob_1.append(dictionary.get("values")[1])
				save_df = pd.DataFrame()
				save_df["prediction"]=pred_spark["prediction"]
				save_df["prob_0"]=prob_0
				save_df["prob_1"]=prob_1
				csvFileName = sparkPath + jsonFolder + ".csv"
				save_df.to_csv(csvFileName, index=False)
				print("%s was created." % csvFileName)

def getDataFromCsv(filename):
	print(filename)
	csv_df = pd.read_csv(filename)
	prediction = csv_df["prediction"]
	prob_0 = csv_df["prob_0"]
	prob_1 = csv_df["prob_1"]
	return prediction, prob_0, prob_1	

def compareByMatchingTable():
	print('MatchingTable:')

	matching_table = pd.read_csv('algorithm-descriptions/framework_matching.csv')
	print(matching_table)

	result_columns = ["same","TP","FP","FN","TN","compared_algorithm"]
	results_df = pd.DataFrame(columns=result_columns)
	print(results_df)
	for index, row in matching_table.iterrows():
		algorithm_ident = row['sklearn']

		print('\nThe predictions in the following files are compared with each other:')
		
		if row['sklearn'] == "None":
			flag_sklearn = False
			algorithm_ident = row['spark']
			
		else:
			algorithm_ident = row['sklearn']

			# get sklearn csv file
			flag_sklearn = True
			sklearn_csvfile = 'predictions/sklearn/pred_SKLEARN_' + row['sklearn'] + '_Uniform.csv'
			pred_sklearn, prob_0_sklearn, prob_1_sklearn = getDataFromCsv(sklearn_csvfile)

		if row['weka'] == "None":
			flag_weka = False
		else:
			# get weka csv file
			flag_weka = True
			weka_csvfile = 'predictions/weka/pred_WEKA_' + row['weka'] + '_Uniform.csv'
			pred_weka, prob_0_weka, prob_1_weka = getDataFromCsv(weka_csvfile)
		

		if row['spark'] == "None":
			flag_spark = False
		else:
			flag_spark = True
			# spark CSVs are currently saved in a weird format (csv file is in a named folder)
			spark_csvfile = 'predictions/spark/pred_SPARK_' + row['spark'] + '_Uniform.csv'
			pred_spark, prob_0_spark, prob_1_spark = getDataFromCsv(spark_csvfile)


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

		if flag_sklearn and flag_spark:
			results_df, _, _=createConfusionMatrix(algorithm_ident, [pred_sklearn, pred_spark], ["sklearn", "spark"], results_df, index)
		if flag_sklearn and flag_weka:
			results_df, _, _=createConfusionMatrix(algorithm_ident, [pred_sklearn, pred_weka], ["sklearn", "weka"], results_df, index)
		if flag_spark and flag_weka:
			results_df, _, _=createConfusionMatrix(algorithm_ident, [pred_spark, pred_weka], ["spark", "weka"], results_df, index)

	print("\n ResultDataFrame:")
	print(results_df)
	results_df.to_csv('algorithm-descriptions/framework_matching_results.csv', index=False)


def compareByName(): # under construction
	print("under construction")


if __name__ == "__main__":
	compareByMatchingTable()
