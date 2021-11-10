"""Set of functions needed for the evaluation of the results.

With a call of the evaluateResults() function the module takes the output of the atoml tests (predictions folder)
and the algorithm-descriptions folder as input. From that it generates result metrics and plots.
The results can be saved in an archive.
"""

import os
import sys
import itertools
from datetime import datetime
import pandas as pd
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import chi2_contingency, ks_2samp
from typing import List

RESULT_DF_COLUMNS = ["delta", "delta_score", "TP", "FP", "FN", "TN", "ks_pval", "chi2_pval",
                     "accs_fw1", "accs_fw2", "algorithm", "framework_1", "framework_2", "dataset", "training_as_test"]
ALPHA_THRESHOLD = 0.05
EPSILON_EQUAL_SCORE = 1e-3

########################################
# CLASSES
########################################


class Algorithm:
    """Holds all information about one specific implementation tested on one specific dataset.

    Attributes:
        filename: csv file name of the data file
        path: path to the csv file folder
        framework: framework of the implementation
        name: name of the implemented algorithm
        dataset_type: name of the dataset the implementation was tested with
        test_as_training: flag for the use of training data as test data
        predictions: predicted labels of the implementation on the data set
        probabilities: tuple of predicted probabilities for both classes of the implementation on the data set
        actuals: actual labels of the data set
    """

    def __init__(self, filename: str, path: str = None, print_all=True):
        self.filename = filename
        self.path = path
        self.framework, self.name, self.dataset_type, self.training_as_test = get_pred_file_metadata(filename)
        self.predictions, self.probabilities, _, self.actuals = get_data_from_csv(
            os.path.join(self.path, self.filename), print_all=print_all)

    def __str__(self):
        return self.framework + ' ' + self.name


class Archive:
    """Instance of an archive where all the generated metrics and plots can be saved.

    The archive folder name is generated with a time stamp by default.
    Alternativly, a name can be given to the constructor.
    In addition to the evaluation results also the foundation can be saved, meaning the current yaml-folder.
    Be aware that this can lead to inconsistencies, when changing the folder's content during runtime.
    Moreover, the predictions can be saved for reproducing the evaluations.

   Attributes:
       path: path of the archive
       print_all (boolean): if flag is set results are printed in the function
   """

    def __init__(self, name: str = None, archive_folder=None, yaml_folder=None,
                 pred_folder=None, test_folder=None, print_all=True):
        self.print_all = print_all

        # create the archive folder
        if not os.path.exists(archive_folder):
            os.makedirs(archive_folder)

        if name is None:
            # create timestamp for the naming of the archive folder
            date_time_obj = datetime.now()
            timestamp = ("%d-%02d-%02d" % (date_time_obj.year, date_time_obj.month, date_time_obj.day))
            timestamp += ("_%02d-%02d" % (date_time_obj.hour, date_time_obj.minute))
            self.path = os.path.join(archive_folder, timestamp)
        else:
            self.path = os.path.join(archive_folder, name)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if yaml_folder is not None:
            if not os.path.exists(os.path.join(self.path, 'yaml_descriptions')):
                os.makedirs(os.path.join(self.path, 'yaml_descriptions'))
            # save the current yml files
            file_list = [f for f in os.listdir(yaml_folder)]
            for file in file_list:
                if file.endswith(".yml") or file.endswith(".yaml"):
                    copyfile(os.path.join(yaml_folder, file), os.path.join(self.path, 'yaml_descriptions', file))

        if test_folder is not None:
            if not os.path.exists(os.path.join(self.path, 'tests')):
                os.makedirs(os.path.join(self.path, 'tests'))
            # save the current generated test files
            for root, dirs, files in os.walk(test_folder):
                for file in files:
                    if file.endswith('.py') or file.endswith('.java') or file.endswith('.R'):
                        copyfile(os.path.join(root, file), os.path.join(self.path, 'tests', file))

        if pred_folder is not None:
            if not os.path.exists(os.path.join(self.path, 'predictions')):
                os.makedirs(os.path.join(self.path, 'predictions'))
            # save the current prediction files
            for root, dirs, files in os.walk(pred_folder):
                for file in files:
                    if file.endswith('.csv'):
                        copyfile(os.path.join(root, file), os.path.join(self.path, 'predictions', file))

        if not os.path.exists(os.path.join(self.path, 'plots')):
            os.makedirs(os.path.join(self.path, 'plots'))

        if not os.path.exists(os.path.join(self.path, 'views_by_dataset')):
            os.makedirs(os.path.join(self.path, 'views_by_dataset'))

        if not os.path.exists(os.path.join(self.path, 'views_by_algorithm')):
            os.makedirs(os.path.join(self.path, 'views_by_algorithm'))

        if self.print_all:
            print("New archive folder created: %s\n" % self.path)

    def archive_data_frame(self, df, filename="result_table.csv", by_dataset=False, by_algorithm=False):
        # save the summary of the evaluation in a csv file
        if by_dataset:
            csv_save_path = os.path.join(self.path, 'views_by_dataset', filename)

        elif by_algorithm:
            csv_save_path = os.path.join(self.path, 'views_by_algorithm', filename)
        else:
            csv_save_path = os.path.join(self.path, filename)
        df.to_csv(csv_save_path, index=False)
        if self.print_all:
            print("\nResult DataFrame saved at: %s" % os.path.join(self.path, filename))


########################################
# CSV FILE TOOLS
########################################


def get_pred_file_metadata(filename: str):
    """Splits a prediction csv filename to get the information it contains.

    Args:
        filename:
            The filename should consist of the keyword 'pred' and 3 identifier:
            for the framework (in capital letters), the algorithm and the used data set type.
            Example: 'pred_FRAMEWORK_Algorithm_TestDataType.csv'

    Returns:
        (str, str, str):
            - Name of the framework
            - Name of the algorithm
            - Name of the dataset
            - Training data as test data flag
    """
    if filename.endswith(".csv"):
        string = filename[:-4]
        substring = string.split("_")
        training_as_test = False
        if substring[-2] == "TrainingAsTest":
            training_as_test = True
        if len(substring) != (training_as_test + 5):
            print("Warning: The prediction filename: %s doesn't consist out the right amount of substrings." % filename)
        framework = substring[1]
        algorithm = substring[2]
        dataset_type = substring[3]
    else:
        print("Error: %s is not a csv file." % filename)
        return
    return framework, algorithm, dataset_type, training_as_test


def get_data_from_csv(filename: str, print_all=True):
    """Reads in a csv file with the specified format and extracts the different columns.

    Args:
        filename (str): relative or absolute filepath
        print_all (boolean): if flag is set results are printed in the function

    Returns: a tuple with 4 variables
        (Series, Series, Series, Series):
            - predicted labels
            - predicted probability for class 0
            - predicted probability for class 1
            - actual labels
    """
    if print_all:
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


def create_confusion_matrix(predictions: [pd.Series, pd.Series], print_all=True):
    """Creates the confusion matrix between 2 sets of labels.

    Args:
        predictions ([Series, Series]): 2 sets of prediction labels
        print_all (boolean): if flag is set results are printed in the function

    Returns:
        (boolean, 2x2 array):
            - Equality flag: True, if predicted labels are identical
            - Confusion matrix
    """
    cm = confusion_matrix(predictions[0], predictions[1])
    # if all the values are the same, the confusion matrix has only one element this is checked and changed here
    if cm.shape == (1, 1):
        num_elements = predictions[0].shape[0]
        if predictions[0][0] == 0:
            cm = np.array([[num_elements, 0], [0, 0]])
        else:
            cm = np.array([[0, 0], [0, num_elements]])
    if print_all:
        print(cm)
    if cm[0, 1] == 0 and cm[1, 0] == 0 and (cm[0, 0] != 0 or cm[1, 1] != 0):
        equal = True
    else:
        equal = False
    return equal, cm


def ks_statistic(probabilities: [pd.Series, pd.Series], print_all=True):
    """Does the Kolmogorov–Smirnov test with two probability distributions.

    Args:
        probabilities ([Series, Series): Two sets of propability distributions
        print_all (boolean): if flag is set results are printed in the function

    Returns:
        (float, float):
            - p-value of KS test
            - KS test statistic
    """
    ks_result = ks_2samp(probabilities[0], probabilities[1])
    p = ks_result.pvalue
    ks_score = ks_result.statistic

    # interpret p-value
    alpha = ALPHA_THRESHOLD
    if print_all:
        print("KS p-value: %0.4f\t ks-statistic: %0.4f" % (p, ks_score))
        if p <= alpha:
            print('Different (reject H0)\n')
        else:
            print('Equal (H0 holds true)\n')
    return p, ks_score


def chi2_statistic(pred: [pd.Series, pd.Series], print_all=True):
    """Does the Chi-squared test with two sets of prediction labels.

    Args:
        pred ([Series, Series): Two sets of prediction labels
        print_all (boolean): if flag is set results are printed in the function

    Returns:
        float: p-value of chi-squared test
    """
    data = [[sum(pred[0]), len(pred[0]) - sum(pred[0])], [sum(pred[1]), len(pred[1]) - sum(pred[1])]]
    try:
        stat, p, dof, expected = chi2_contingency(data)
        # interpret p-value
        alpha = ALPHA_THRESHOLD
        if print_all:
            print("Chi² p-value: %0.4f" % p)
            if p <= alpha:
                print('Dependent (reject H0)\n')
            else:
                print('Independent (H0 holds true)\n')
    except ValueError:
        print("Chi-squared test failed, probably all elements are identical")
        p = 0.0
    return p


def get_delta_of_scores(pred_prob1: pd.Series, pred_prob2:pd.Series) -> int:
    """Compares the scores (pred_prob) of two algorithms and calculates a delta value.

    Args:
        pred_prob1: prediction probabilities (scores) of the first algorithm
        pred_prob2: prediction probabilities (scores) of the second algorithm

    Returns:
        number of results where the difference between the scores is greater than a defined epsilon
    """
    delta_score = 0
    for score1, score2 in zip(pred_prob1, pred_prob2):
        if abs(score1 - score2) > EPSILON_EQUAL_SCORE:
            delta_score += 1
    return delta_score


def compare_two_algorithms(x: Algorithm, y: Algorithm, df: pd.DataFrame, print_all=True):
    """Compares two prediction results and creates different metrics.
    The metrics are the confusion matrix, the Kolmogorov–Smirnov test result, the Chi2 test result and also the
    accuracy of the two prediction sets compared to the actual values

    Args:
        x: the results for one specific algorithm implementation on one dataset
        y: the results for one specific algorithm implementation on one dataset
        df: result overview dataframe with different metrics
        print_all (boolean): if flag is set results are printed in the function

    Returns:
        DataFrame: result overview dataframe with different metrics
    """
    if print_all:
        print(f'Comparison between {x.framework.upper()} and {y.framework.upper()} for {x.name}')

    df = df.append(pd.Series(dtype="object"), ignore_index=True)
    df.loc[df.index[-1], 'algorithm'] = x.name
    df.loc[df.index[-1], 'framework_1'] = x.framework
    df.loc[df.index[-1], 'framework_2'] = y.framework
    df.loc[df.index[-1], 'dataset'] = x.dataset_type
    df.loc[df.index[-1], 'training_as_test'] = x.training_as_test

    _, cm = create_confusion_matrix([x.predictions, y.predictions], print_all)
    df.loc[df.index[-1], 'TP'] = cm[0, 0]
    df.loc[df.index[-1], 'FP'] = cm[0, 1]
    df.loc[df.index[-1], 'FN'] = cm[1, 0]
    df.loc[df.index[-1], 'TN'] = cm[1, 1]
    df.loc[df.index[-1], 'delta'] = cm[0, 1] + cm[1, 0]
    df.loc[df.index[-1], 'delta_score'] = get_delta_of_scores(x.probabilities, y.probabilities)

    ks_pval, _ = ks_statistic([x.probabilities, y.probabilities], print_all)
    df.loc[df.index[-1], 'ks_pval'] = ("%0.3f" % ks_pval)

    chi2_pval = chi2_statistic([x.predictions, y.predictions], print_all)
    df.loc[df.index[-1], 'chi2_pval'] = ("%0.3f" % chi2_pval)

    accs_fw_i = accuracy_score(x.actuals, x.predictions)
    accs_fw_j = accuracy_score(y.actuals, y.predictions)
    if print_all:
        print("%s accuracy: %f" % (x.framework, accs_fw_i))
        print("%s accuracy: %f\n" % (y.framework, accs_fw_j))
    df.loc[df.index[-1], 'accs_fw1'] = ("%0.3f" % accs_fw_i)
    df.loc[df.index[-1], 'accs_fw2'] = ("%0.3f" % accs_fw_j)
    return df


########################################
# EVALUATE RESULTS
########################################

def set_pandas_print_full_df():
    """Sets the pandas options to print a full dataframe without cutting of parts."""
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


def plot_probabilities(algorithms: List[Algorithm], archive: Archive = None, show_plot=False, print_all=True):
    """Plots the probability distributions of a list of implementations.

    Args:
        algorithms: list of implementation instances which probabilities are to plot
        archive: archive instance which is used to save data
        show_plot: if flag is set the plot is shown in program
        print_all (boolean): if flag is set results are printed in the function
    """
    if not algorithms:
        print("Nothing to plot. The algorihm list is empty.\n")
    else:
        try:
            plt.figure()
            if print_all:
                print("plot probabilities for %s with %s dataset\n" % (algorithms[0].name, algorithms[0].dataset_type))
            num_bins = 20
            if algorithms[0].probabilities.shape[0] > 500:
                num_bins = 100
            for x in algorithms:
                sns.distplot(x.probabilities, kde=False, label=x.framework, hist_kws=dict(alpha=0.3), bins=num_bins)
            if algorithms[0].training_as_test:
                plt.title(("prediction probability of " + algorithms[0].name +
                           "\non " + algorithms[0].dataset_type + " dataset with training data as test data"))
            else:
                plt.title(("prediction probability of " + algorithms[0].name +
                           "\non " + algorithms[0].dataset_type + " dataset"))
            plt.legend()

            if show_plot:
                plt.show()

            if archive is not None:
                if algorithms[0].training_as_test:
                    plot_file_name = "TaT_" + algorithms[0].dataset_type + '_' + algorithms[0].name + "_probabilities.svg"
                    plt.savefig(os.path.join(archive.path, "plots", plot_file_name))
                else:
                    plot_file_name = algorithms[0].dataset_type + '_' + algorithms[0].name + "_probabilities.svg"
                    plt.savefig(os.path.join(archive.path, "plots", plot_file_name))
            plt.close()
        except:
            print(sys.exc_info()[0], " while printing probability predictions of %s on %s data.\n"
                  % (algorithms[0].name, algorithms[0].dataset_type))


def create_views_by_algorithm(df: pd.DataFrame = None, csv_file: str = None, archive: Archive = None, print_all=True):
    """Creates a views on dataframe based on the algorithms.

    The function creates smaller dataframes only containing comparison results based on the same algorithm out of a
    bigger dataframe. It works either direcly with a DataFrame as input or with the path of a csv file containing the
    input dataframe. If both are given the DataFrame is used. The result can be shown and/or saved in an archive.

    Args:
        df: DataFrame from which to create views for the single algorithms
        csv_file: path to the csv file with the Dataframe from which to create views for the single algorithms
        archive: archive instance which is used to save data
        print_all: (boolean): if flag is set results are printed in the function

    Raises:
        RuntimeError: if neither Dataframe nor csv file are given as input.
    """
    if df is not None:
        pass
    elif csv_file is not None:
        df = pd.read_csv(csv_file)
    else:
        msg = "Cannot create a view without an input Dataframe or csv file being specified."
        raise (RuntimeError(msg))

    if print_all:
        set_pandas_print_full_df()

    for alg in df.algorithm.unique():
        df_view = df.loc[df.algorithm == alg]
        if print_all:
            print(f"\n{alg}")
            print(df_view)
        if archive is not None:
            archive.archive_data_frame(df_view, filename=(alg + "_cmp_results.csv"), by_algorithm=True)


def evaluate_results(prediction_folder: str, yaml_folder: str = None, archive_folder: str = None,
                     gen_tests_folder: str = None, print_all=True):
    """Main function for the evaluation of the prediction csv files.

    The function reads in all csv files from a specific folder. Gathers meta data from the csv file names
    and evaluates the content of the files. For that it creates different metrics and histograms which can be saved
    together with the current yaml folder for the csv file creation in an archive folder.

    Args:
        prediction_folder: relative path to the folder with the prediction files
        yaml_folder: relative path to the folder with the yaml definitions of the ML algorithms. If no folder is given,
            the yaml files will not be saved in the archive.
        archive_folder: relative path to the folder where the archive should be saved. If no folder is given,
            no archive will be created.
        gen_tests_folder: relative path to the folder where the test cases are located. This is only for the archiving.
            If no folder is given, the tests will not be stored in archive.
        print_all (boolean): if flag is set results are printed in the function
    """
    set_pandas_print_full_df()

    if archive_folder is not None:
        archive = Archive(archive_folder=archive_folder, yaml_folder=yaml_folder, pred_folder=prediction_folder,
                          test_folder=gen_tests_folder, print_all=print_all)
    else:
        archive = None
    algorithm_list = []
    num_csv_files_read = 0
    # read prediction files and save data for every algorithm implementation in a Algorithm class
    framework_list = [fw for fw in os.listdir(prediction_folder)]
    for fw in framework_list:
        csv_list = [f for f in os.listdir(os.path.join(prediction_folder, fw))]
        for file in csv_list:
            if file.endswith(".csv"):
                algorithm_list.append(Algorithm(file, os.path.join(prediction_folder, fw), print_all=print_all))
                num_csv_files_read += 1

    # get all types of algorithm (unique_algorithm_list) and all types of datasets
    dataset_list = []
    unique_algorithm_list = []
    for algorithm in algorithm_list:
        if algorithm.dataset_type not in dataset_list:
            dataset_list.append(algorithm.dataset_type)
        if algorithm.name not in unique_algorithm_list:
            unique_algorithm_list.append(algorithm.name)
    if print_all:
        print("\nList of unique algorithm identifier:\n", unique_algorithm_list, "\n")

    overall_results_df = pd.DataFrame(columns=RESULT_DF_COLUMNS)

    for ds in dataset_list:
        print(f"processing predictions on {ds}.")
        # create a dataframe for the results
        dataset_results_df = pd.DataFrame(columns=RESULT_DF_COLUMNS)

        algorithm_subset_by_dataset_without_TAT = [x for x in algorithm_list
                                                   if (x.dataset_type == ds) & (x.training_as_test is False)]
        algorithm_subset_by_dataset_with_TAT = [x for x in algorithm_list
                                                if (x.dataset_type == ds) & (x.training_as_test is True)]
        for alg in unique_algorithm_list:
            algorithm_subset = [x for x in algorithm_subset_by_dataset_without_TAT if x.name == alg]
            algorithm_subset_TAT = [x for x in algorithm_subset_by_dataset_with_TAT if x.name == alg]

            if algorithm_subset:
                plot_probabilities(algorithm_subset, archive, print_all=print_all)
                for a, b in itertools.combinations(algorithm_subset, 2):
                    dataset_results_df = compare_two_algorithms(a, b, dataset_results_df, print_all=print_all)

            if algorithm_subset_TAT:
                plot_probabilities(algorithm_subset_TAT, archive, print_all=print_all)
                for a, b in itertools.combinations(algorithm_subset_TAT, 2):
                    dataset_results_df = compare_two_algorithms(a, b, dataset_results_df, print_all=print_all)

        overall_results_df = overall_results_df.append(dataset_results_df)

        if print_all:
            print("\nSummary DataFrame for %s dataset:" % ds)
            print(dataset_results_df)

        if archive_folder is not None:
            archive.archive_data_frame(dataset_results_df, filename=(ds + "_cmp_results.csv"), by_dataset=True)

    if archive_folder is not None:
        archive.archive_data_frame(overall_results_df, filename="ALL_cmp_results.csv")

    create_views_by_algorithm(df=overall_results_df, archive=archive, print_all=True)

    return num_csv_files_read


if __name__ == "__main__":
    print(sys.argv)
    evaluate_results(prediction_folder="predictions", yaml_folder="algorithm-descriptions",
                     archive_folder="archive", gen_tests_folder="generated-tests", print_all=False)
