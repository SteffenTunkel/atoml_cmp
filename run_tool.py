"""This module runs the tool"""

import os
import shutil
import evaluation
import build_docker
from external_data_utils import overwrite_dataset, rename_prediction_file
from distutils.dir_util import copy_tree

# cmd /k -> remain ___ cmd /c -> terminate
# Example for a system call for a docker with a mount:
# os.system('cmd /c "docker run --mount type=bind,source=%cd%/algorithm-descriptions,target=/testdata atoml_docker"')


def run_my_docker(name, *bindings, option=None):
    """
    Starts a docker container.
    The parameter name includes the container image name, which needs to be build before.
    The parameter bindings is a list of string couples. It contains the mount bindings with the source relative
    to the current dictionary as the first string and the target as the second
    """
    # build command string
    cmdstr = 'cmd /c"docker run '
    for bind in bindings:
        cmdstr += '--mount type=bind,source=%cd%/'
        cmdstr += bind[0]
        cmdstr += ',target='
        cmdstr += bind[1]
        cmdstr += ' '
    cmdstr += name
    if option is not None:
        cmdstr += ' ' + str(option)
    cmdstr += '"'
    # run command
    os.system(cmdstr)


def delete_folder(name):
    try:
        shutil.rmtree(name)
        print("successfully deleted: %s" % name)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


# clear generated-tests and predictions folder
delete_folder("generated-tests")
delete_folder("predictions")

# run atoml docker build
#os.system('cmd /c "docker build -t atoml_docker atoml_docker"')  # --no-cache

atomlMounts = [["generated-tests", "/container/generated-tests"], ["algorithm-descriptions", "/container/testdata"]]
run_my_docker("atoml_docker", *atomlMounts, option=100)

overwrite_datasets = [["BreastCancer", "Zeroes"], ["BreastCancerMinMaxNorm", "VerySmall"], ["BreastCancerMeanNorm", "Bias"],
                      ["Wine", "LeftSkew"], ["WineMinMaxNorm", "RightSkew"], ["WineMeanNorm", "Outlier"]]
for overwrite_pair in overwrite_datasets:
    overwrite_dataset(overwrite_pair[0]+"_1_training.arff", overwrite_pair[1]+"_1_training.arff")
    overwrite_dataset(overwrite_pair[0]+"_1_test.arff", overwrite_pair[1]+"_1_test.arff")


sklearnMounts = [["generated-tests/sklearn", "/container/generated-tests/sklearn"], ["predictions/sklearn", "/container/predictions"]]
run_my_docker("sklearn_docker", *sklearnMounts)

caretMounts = [["generated-tests/caret", "/container/generated-tests/caret"], ["predictions/caret", "/container/predictions"]]
run_my_docker("caret_docker", *caretMounts)

wekaMounts = [["generated-tests/weka/src", "/container/src"], ["predictions/weka", "/container/predictions"]]
run_my_docker("weka_docker", *wekaMounts)

sparkMounts = [["generated-tests/spark/src", "/container/src"], ["predictions/spark", "/container/predictions"]]
run_my_docker("spark_docker", *sparkMounts)

for overwrite_pair in overwrite_datasets:
    rename_prediction_file(overwrite_pair[0], overwrite_pair[1])

evaluation.evaluate_results(print_all=False)
