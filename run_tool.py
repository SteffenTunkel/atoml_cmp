"""This module runs the tool"""

import os
import shutil
import evaluation
from distutils.dir_util import copy_tree
import build_docker

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
run_my_docker("atoml_docker", *atomlMounts, option=200)

sklearnMounts = [["generated-tests/sklearn", "/container/generated-tests/sklearn"], ["predictions/sklearn", "/container/predictions"]]
run_my_docker("sklearn_docker", *sklearnMounts)

caretMounts = [["generated-tests/caret", "/container/generated-tests/caret"], ["predictions/caret", "/container/predictions"]]
run_my_docker("caret_docker", *caretMounts)

wekaMounts = [["generated-tests/weka/src", "/container/src"], ["predictions/weka", "/container/predictions"]]
run_my_docker("weka_docker", *wekaMounts)

sparkMounts = [["generated-tests/spark/src", "/container/src"], ["predictions/spark", "/container/predictions"]]
run_my_docker("spark_docker", *sparkMounts)

evaluation.evaluate_results(print_all=False)
