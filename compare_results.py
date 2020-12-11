# at the moment limited to the Uniform data

import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_1samp, ks_2samp

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

def KS_statistic(algorithm_identifier, predictions, frameworks, df, iteration):
		print("under construction")

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
	
	# get the information from the matching table
	print('MatchingTable:')
	matching_table = pd.read_csv('algorithm-descriptions/framework_matching.csv')
	print(matching_table)
	frameworks = matching_table.columns

	# create a dataframe for the results
	result_columns = ["same","TP","FP","FN","TN","compared_algorithm"]
	results_df = pd.DataFrame(columns=result_columns)

	for index, row in matching_table.iterrows():
		framework_flags	= [False] * len(frameworks)
		predictions		= [None] * len(frameworks)
		prob_0 			= [None] * len(frameworks)
		prob_1 			= [None] * len(frameworks)

		# open the prediction files
		print('\nThe predictions in the following files are compared with each other:')
		for i, framework in enumerate(frameworks):
			if row[framework] != "None":
				framework_flags[i] = True
				algorithm_ident = row[framework]

				csvfile = 'predictions/' + framework
				csvfile += '/pred_' + framework.upper() + '_' + row[framework] + '_Uniform.csv'
				predictions[i], prob_0[i], prob_1[i] = getDataFromCsv(csvfile)

		# comparison between the frameworks
		print('')
		for i in range(len(frameworks)):
			for j in range(len(frameworks)):
				if i < j:
					if framework_flags[i] and framework_flags[j]:
						results_df, _, _=createConfusionMatrix(algorithm_ident, [predictions[i], predictions[j]],
																[frameworks[i], frameworks[j]], results_df, index)
						ks_result = ks_2samp(prob_0[i], prob_0[j])
						print("KS: p-value=%0.4f, ks-statistic=%0.4f" % (ks_result.pvalue, ks_result.statistic))
						print('')

	# store results
	print("\n ResultDataFrame:")
	print(results_df)
	results_df.to_csv('algorithm-descriptions/framework_matching_results.csv', index=False)


def compareByName(): # under construction
	print("under construction")


if __name__ == "__main__":
	compareByMatchingTable()
	
