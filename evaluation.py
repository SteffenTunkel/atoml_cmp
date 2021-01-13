"""
Set of functions needed for the evaluation of the results.
Takes the output of the atoml tests (predictions folder)
    and the algorithm-descriptions folder as input.
The main function is 'evaluateResults()'.
"""

import os
import itertools
from datetime import datetime
import pandas as pd
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import chi2_contingency, ks_2samp


YAML_FOLDER = 'algorithm-descriptions'
PREDICTION_FOLDER = 'predictions'
ARCHIVE_FOLDER = 'archive'
BASH_FILE = "docker_run_atoml_testgeneration.sh"
RESULT_DF_COLUMNS = ["equal_pred", "TP", "FP", "FN", "TN", "ks_pval", "chi2_pval",
                     "accs_fw1", "accs_fw2", "algorithm", "parameters"]


########################################
# CLASSES
########################################


class Algorithm:
    """missing doc"""

    def __init__(self, filename, path=None):
        self.filename = filename
        self.path = path
        self.framework, self.name, self.dataset_type = split_prediction_file(filename)
        self.predictions, self.probabilities, _, self.actuals = get_data_from_csv(
            os.path.join(self.path, self.filename))

    def __str__(self):
        return self.framework + ' ' + self.name


class Archive:
    """missing doc"""

    def __init__(self, name=None, save_yaml=True, save_bash=False):
        # create the archive folder
        if not os.path.exists(ARCHIVE_FOLDER):
            os.makedirs(ARCHIVE_FOLDER)

        if name is None:
            # create timestamp for the naming of the archive folder
            date_time_obj = datetime.now()
            timestamp = ("%d-%02d-%02d" % (date_time_obj.year, date_time_obj.month, date_time_obj.day))
            timestamp += ("_%02d-%02d" % (date_time_obj.hour, date_time_obj.minute))
            self.path = os.path.join(ARCHIVE_FOLDER, timestamp)
        else:
            self.path = os.path.join(ARCHIVE_FOLDER, name)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if save_yaml is True:
            if not os.path.exists(os.path.join(self.path, 'yaml_descriptions')):
                os.makedirs(os.path.join(self.path, 'yaml_descriptions'))

            # save the current yml files
            file_list = [f for f in os.listdir(YAML_FOLDER)]
            for file in file_list:
                if file.endswith(".yml") or file.endswith(".yaml"):
                    copyfile(os.path.join(YAML_FOLDER, file), os.path.join(self.path, 'yaml_descriptions', file))

        # save the call function for the atoml test generation, not needed if the whole yml folder is run automatically
        if save_bash is True:
            copyfile(os.path.join("atoml_docker", BASH_FILE), os.path.join(self.path, BASH_FILE))

        print("\nNew archive folder created: %s" % self.path)

    def archive_data_frame(self, df, filename="result_table.csv"):
        # save the summary of the evaluation in a csv file
        df.to_csv(os.path.join(self.path, filename), index=False)
        print("\nResult DataFrame saved at: %s" % os.path.join(self.path, filename))


########################################
# CSV FILE TOOLS
########################################


def split_prediction_file(filename):
    """
    Splits the name a csv file to get the information it contains. The filename should consist of the keyword 'pred'
    and 3 identifier, for the framework (in capital letters), the algorithm and the used data set type.
    All of them should be divided by exactly one '_'. Example: 'pred_FRAMEWORK_Algorithm_TestDataType.csv'.
    """

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


def get_data_from_csv(filename):
    """Reads in a csv file with certain format and extracts the different columns."""

    print("read: %s" % filename)
    csv_df = pd.read_csv(filename)
    actual = csv_df["actual"]
    prediction = csv_df["prediction"]
    prob_0 = csv_df["prob_0"]
    prob_1 = csv_df["prob_1"]
    return prediction, prob_0, prob_1, actual


########################################
# CALCULATE METRICS
########################################


def create_confusion_matrix(predictions):
    """missing doc"""

    cm = confusion_matrix(predictions[0], predictions[1])

    # if all the values are the same, the confusion matrix has only one element
    # this is checked and changed here
    if cm.shape == (1, 1):
        num_elements = predictions[0].shape[0]
        if predictions[0][0] == 0:
            cm = np.array([[num_elements, 0], [0, 0]])
        else:
            cm = np.array([[0, 0], [0, num_elements]])

    print(cm)

    if cm[0, 1] == 0 and cm[1, 0] == 0 and (cm[0, 0] != 0 or cm[1, 1] != 0):
        equal = True
    else:
        equal = False
    return equal, cm


def ks_statistic(probabilities):
    """missing doc"""

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
    """missing doc"""

    # get data values
    data = [[sum(pred[0]), len(pred[0]) - sum(pred[0])], [sum(pred[1]), len(pred[1]) - sum(pred[1])]]
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


def compare_two_algorithms(x, y, df, i):
    """missing doc"""

    print('%d: Comparision between %s and %s for %s' % (i, x.framework.upper(), y.framework.upper(), x.name))

    df = df.append(pd.Series(), ignore_index=True)
    df.loc[df.index[-1], 'algorithm'] = x.name
    df.loc[df.index[-1], 'parameters'] = ('%s_%s_%s' % (x.framework, y.framework, x.dataset_type))

    equal_pred, cm = create_confusion_matrix([x.predictions, y.predictions])
    df.loc[df.index[-1], 'equal_pred'] = equal_pred
    df.loc[df.index[-1], 'TP'] = cm[0, 0]
    df.loc[df.index[-1], 'FP'] = cm[0, 1]
    df.loc[df.index[-1], 'FN'] = cm[1, 0]
    df.loc[df.index[-1], 'TN'] = cm[1, 1]

    ks_pval, _ = ks_statistic([x.probabilities, y.probabilities])
    df.loc[df.index[-1], 'ks_pval'] = ("%0.3f" % ks_pval)

    chi2_pval = chi2_statistic([x.predictions, y.predictions])
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
# EVALUATE RESULTS
########################################


def plot_probabilities(algorithms, archive=None, show_plot=False):
    """missing doc"""
    plt.figure()
    for x in algorithms:
        sns.distplot(x.probabilities, label=x.framework, hist_kws={'alpha': 0.5})
    plt.title(algorithms[0].name)
    plt.legend()

    if show_plot:
        plt.show()

    if archive is not None:
        plot_file_name = algorithms[0].dataset_type + '_' + algorithms[0].name + "_probabilities.pdf"
        plt.savefig(os.path.join(archive.path, plot_file_name))


def evaluate_results():
    """missing doc"""
    # get list of files for all frameworks
    # list all csv files. compare them
    archive = Archive()
    algorithm_list = []

    framework_list = [fw for fw in os.listdir(PREDICTION_FOLDER)]

    for fw in framework_list:
        csv_list = [f for f in os.listdir(os.path.join(PREDICTION_FOLDER, fw))]
        for file in csv_list:
            if file.endswith(".csv"):
                algorithm_list.append(Algorithm(file, os.path.join(PREDICTION_FOLDER, fw)))

    dataset_list = []
    unique_algorithm_list = []

    for algorithm in algorithm_list:
        if algorithm.dataset_type not in dataset_list:
            dataset_list.append(algorithm.dataset_type)
        if algorithm.name not in unique_algorithm_list:
            unique_algorithm_list.append(algorithm.name)
    print("\nList of unique algorithm identifier (for typo check):")
    print(unique_algorithm_list)
    print('')

    # create a dataframe for the results
    results_df = pd.DataFrame(columns=RESULT_DF_COLUMNS)
    i = 0

    for ds in dataset_list:
        algorithm_subset_by_dataset_type = [x for i, x in enumerate(algorithm_list) if x.dataset_type == ds]
        for alg in unique_algorithm_list:
            algorithm_subset = [x for i, x in enumerate(algorithm_subset_by_dataset_type) if x.name == alg]
            plot_probabilities(algorithm_subset, archive)
            for a, b in itertools.combinations(algorithm_subset, 2):
                results_df = compare_two_algorithms(a, b, results_df, i)
                i = i + 1

        # set pandas options to print a full dataframe
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)

        # show summary data frame
        print("\nSummary DataFrame for %s dataset:" % ds)
        print(results_df)

        archive.archive_data_frame(results_df, filename=(ds + "_result_summary.csv"))


if __name__ == "__main__":
    evaluate_results()
