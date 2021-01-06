'''
Set of functions needed for the evaluation of the results.
Takes the output of the atoml tests (predctions folder)
	and the algorithm-descriptions folder as input.
The main function is 'evaluateResults()'.
'''

import os
import itertools
from datetime import datetime
import pandas as pd
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import ttest_1samp, ks_2samp
from scipy.stats import chi2_contingency 

YAML_FOLDER = 'algorithm-descriptions/'
PREDICTION_FOLDER = 'predictions/'
ARCHIVE_FOLDER = 'archive/'
RESULT_DF_COLUMNS = ["equal_pred","TP","FP","FN","TN","ks_pval","chi2_pval","accs_fw1","accs_fw2","algorithm", "parameters"]



########################################
### CLASSES
########################################


class Algorithm:
	'''missing doc'''
	def __init__(self, filename, path=None):
		self.filename = filename
		self.path = path
		self.framework, self.name, self.datasetType = splitPredictionFile(filename)
		self.predictions, self.probabilities, _, self.actuals = getDataFromCsv(self.path + self.filename)
	def __str__(self):
		return self.framework + ' ' +  self.name



class Archive:
	'''missing doc'''
	def __init__(self, name=None, save_yaml=True):
		# create the archive folder
		if not os.path.exists(ARCHIVE_FOLDER):
			os.makedirs(ARCHIVE_FOLDER)
		
		if name is None:
			# create timestamp for the naming of the archive folder
			dateTimeObj = datetime.now()
			timestamp = ("%d-%02d-%02d" % (dateTimeObj.year, dateTimeObj.month, dateTimeObj.day))
			timestamp += ("_%02d-%02d/" % (dateTimeObj.hour, dateTimeObj.minute))
			self.path = ARCHIVE_FOLDER + timestamp
		else:
			if name.endswith("/") or name.endswith('\\'):
				self.path = ARCHIVE_FOLDER + name
			else:
				self.path = ARCHIVE_FOLDER + name + '/'
		
		if not os.path.exists(self.path):
			os.makedirs(self.path)

		if save_yaml is True:
			if not os.path.exists(self.path + 'yaml_descriptions'):
				os.makedirs(self.path + 'yaml_descriptions')

			# save the current yml files
			fileList = [f for f in os.listdir(YAML_FOLDER)]
			for file in fileList:
				if file.endswith(".yml") or file.endswith(".yaml"):
					copyfile(YAML_FOLDER + file, self.path + 'yaml_descriptions/' + file)

		# save the call function for the atoml test generation, not needed when the whole yml folder is run automatically
		#copyfile("atoml_docker/docker_run_atoml_testgeneration.sh", self.path + "docker_run_atoml_copy.sh")

		print("\nNew archive folder created: %s" % self.path)


	def archiveDataframe(self, df, filename="result_table.csv"):
		# save the summary of the evaluation in a csv file
		df.to_csv((self.path + filename), index=False)
		print("\nResult dataframe saved at: %s%s" % (self.path, filename))



########################################
### CSV FILE TOOLS
########################################


def jsonToCsvForSpark():
	'''Converts the json files created by the Spark tests into the csv format, which is used for the evaluation.'''

	sparkPath = PREDICTION_FOLDER + 'spark/'
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
				save_df["actual"]=pred_spark["classAtt"]
				save_df["prediction"]=pred_spark["prediction"]
				save_df["prob_0"]=prob_0
				save_df["prob_1"]=prob_1
				csvFileName = sparkPath + jsonFolder + ".csv"
				save_df.to_csv(csvFileName, index=False)
				print("%s was created." % csvFileName)



def splitPredictionFile(filename):
	'''
	Splits the name a csv file to get the information it contains. The filename should consist of the keyword 'pred'
	and 3 identifier, for the framework (in capital letters), the algorithm and the used dataset type.
	All of them should be devided by exactly one '_'. Example: 'pred_FRAMEWORK_Algorithm_TestDataType.csv'.
	'''

	if filename.endswith(".csv"):
		string = filename[:-4]
		substring = string.split("_")
		if len(substring) != 4:
			print("Error The prediction filename: %s doesn't consist out the right amount of substrings" % filename)
			return
		framework = substring[1]
		algorithm = substring[2]
		dataset_type = substring[3]
	else:
		print("Error: %s is not a csv file." % filename)
		return
	return framework, algorithm, dataset_type



def getDataFromCsv(filename):
	'''Reads in a csv file with certain format and extracts the different columns.'''

	print("read: %s" % filename)
	csv_df = pd.read_csv(filename)
	actual = csv_df["actual"]
	prediction = csv_df["prediction"]
	prob_0 = csv_df["prob_0"]
	prob_1 = csv_df["prob_1"]
	return prediction, prob_0, prob_1, actual



########################################
### CALCULATE METRICS
########################################


def createConfusionMatrix(predictions):
	'''missing doc'''

	cm = confusion_matrix(predictions[0], predictions[1])

	# if all the values are the same, the confusion matrix has only one element
	# this is checked and changed here
	if cm.shape == (1,1):
		num_elements = predictions[0].shape[0]
		if predictions[0][0] == 0:
			cm = np.array([[num_elements,0],[0,0]])
		else:
			cm = np.array([[0,0],[0,num_elements]])

	print(cm)

	if cm[0,1] == 0 and cm[1,0] == 0 and ( cm[0,0] != 0 or cm[1,1] != 0):
		equal = True
	else:
		equal = False
	return equal, cm



def KS_statistic(probabilities):
	'''missing doc'''

	ks_result = ks_2samp(probabilities[0], probabilities[1])
	p = ks_result.pvalue
	ks_score = ks_result.statistic
						
	# interpret p-value 
	alpha = 0.05
	print("KS p-value: %0.4f\t ks-statistic: %0.4f" % (p, ks_score)) 
	if p <= alpha: 
	    print('Different (reject H0)\n') 
	else: 
	    print('Equal (H0 holds true)\n') 
	return p, ks_score



def chi2_statistic(pred):
	'''missing doc'''

	# get data values
	data = [[sum(pred[0]), len(pred[0])-sum(pred[0])] , [sum(pred[1]), len(pred[1])-sum(pred[1])]]
	print(data)
	stat, p, dof, expected = chi2_contingency(data) 
	  
	# interpret p-value 
	alpha = 0.05
	print("Chi² p-value: %0.4f" % p) 
	if p <= alpha: 
	    print('Dependent (reject H0)\n') 
	else: 
	    print('Independent (H0 holds true)\n') 
	return p



def compareTwoAlgorithms(x, y, df, i):
	'''missing doc'''

	print('%d: Comparation between %s and %s for %s' % (i, x.framework.upper(), y.framework.upper(), x.name))

	df = df.append(pd.Series(), ignore_index=True)
	df.loc[df.index[-1], 'algorithm'] = x.name
	df.loc[df.index[-1], 'parameters'] = ('%s_%s_%s' % (x.framework, y.framework, x.datasetType))

	equal_pred, cm = createConfusionMatrix([x.predictions, y.predictions])
	df.loc[df.index[-1], 'equal_pred'] = equal_pred
	df.loc[df.index[-1], 'TP'] = cm[0,0]
	df.loc[df.index[-1], 'FP'] = cm[0,1]
	df.loc[df.index[-1], 'FN'] = cm[1,0]
	df.loc[df.index[-1], 'TN'] = cm[1,1]

	ks_pval, _ = KS_statistic([x.probabilities, y.probabilities])
	df.loc[df.index[-1], 'ks_pval'] = ("%0.3f" % ks_pval)

	chi2_pval= chi2_statistic([x.predictions, y.predictions])
	df.loc[df.index[-1], 'chi2_pval'] = ("%0.3f" % chi2_pval)

	accs_fw_i = accuracy_score(x.actuals, x.predictions)
	accs_fw_j = accuracy_score(y.actuals, y.predictions)
	print("%s accuracy: %f" % (x.framework, accs_fw_i))
	print("%s accuracy: %f" % (y.framework, accs_fw_j))
	df.loc[df.index[-1], 'accs_fw1'] = ("%0.3f" % accs_fw_i)
	df.loc[df.index[-1], 'accs_fw2'] = ("%0.3f" % accs_fw_j)
	print('')

	return df



########################################
### EVALUATE RESULTS
########################################


def plotProbabilities(algorithms, archive=None, show_plot=False):
	'''missing doc'''
	plt.figure()
	for x in algorithms:
		sns.distplot(x.probabilities, label=x.framework, hist_kws={'alpha':0.5})
	plt.title(x.name)
	plt.legend()
	
	if show_plot:
		plt.show()

	if archive is not None:
		plt.savefig(archive.path + algorithms[0].name + "_probabilities.pdf")



def evaluateResults():
	'''missing doc'''
	# get list of files for all frameworks
	# list all csv files. compare them
	archive = Archive()
	algorithmList = []


	frameworkList = [fw for fw in os.listdir(PREDICTION_FOLDER)]

	for fw in frameworkList:
		csv_list = [f for f in os.listdir(PREDICTION_FOLDER + fw)]
		for file in csv_list:
			if file.endswith(".csv"):
				algorithmList.append(Algorithm(file, (PREDICTION_FOLDER + fw + '/')))
	
	datasetList = []
	uniqueAlgorithmList = []

	for algorithm in algorithmList:
		if not algorithm.datasetType in datasetList:
			datasetList.append(algorithm.datasetType)
		if not algorithm.name in uniqueAlgorithmList:
			uniqueAlgorithmList.append(algorithm.name)
	print("\nList of unique algorithm identifier (for typo check):")
	print(uniqueAlgorithmList)
	print('')

	# create a dataframe for the results
	results_df = pd.DataFrame(columns=RESULT_DF_COLUMNS)
	i = 0

	for ds in datasetList:
		algorithmSubsetByDatasetType = [x for i, x in enumerate(algorithmList) if x.datasetType == ds]
		for alg in uniqueAlgorithmList:
			algorithmSubset = [x for i, x in enumerate(algorithmSubsetByDatasetType) if x.name == alg]
			plotProbabilities(algorithmSubset, archive)
			for a, b in itertools.combinations(algorithmSubset, 2):
			    results_df = compareTwoAlgorithms(a, b, results_df, i)
			    i = i + 1

	# set pandas options to print a full dataframe
	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', None)
	pd.set_option('display.max_colwidth', -1)

	# show summary data frame
	print("\nSummaryDataFrame:")
	print(results_df)
	
	# archive summary dataframe and other files, if function is run by another script
	#if __name__ != "__main__":
	archive.archiveDataframe(results_df)



def evaluateResultsWithMatchingTable():
	'''missing doc and outdated at the moment'''

	# get the information from the matching table
	print('MatchingTable:')
	matching_table = pd.read_csv('algorithm-descriptions/framework_matching.csv')
	print(matching_table)
	frameworks = matching_table.columns

	# create a dataframe for the results
	result_columns = RESULT_DF_COLUMNS
	results_df = pd.DataFrame(columns=result_columns)

	for index, row in matching_table.iterrows():
		algorithm_ident = None
		framework_flags	= [False] * len(frameworks)
		predictions		= [None] * len(frameworks)
		prob_0 			= [None] * len(frameworks)
		prob_1 			= [None] * len(frameworks)
		actuals			= [None] * len(frameworks)
		algorithmList = []

		# open the prediction files
		print('\nThe predictions in the following files are compared with each other:')
		for i, framework in enumerate(frameworks):
			if row[framework] != "None":
				framework_flags[i] = True

				if algorithm_ident == None:
					algorithm_ident = row[framework]

				csvpath = PREDICTION_FOLDER + framework + '/'
				csvfile = '/pred_' + framework.upper() + '_' + row[framework] + '_Uniform.csv'
				algorithmList.append(Algorithm(csvfile, csvpath))

		# comparison between the frameworks
		for a, b in itertools.combinations(algorithmList, 2):
		    results_df = compareTwoAlgorithms(a, b, results_df, index)


	# set pandas options to print a full dataframe
	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', None)
	pd.set_option('display.max_colwidth', -1)

	# store results
	print("\nSummaryDataFrame:")
	print(results_df)
	results_df.to_csv('algorithm-descriptions/framework_matching_results.csv', index=False)



if __name__ == "__main__":
	evaluateResults()

