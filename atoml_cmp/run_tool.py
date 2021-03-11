"""This module runs the tool"""

from external_data_utils import overwrite_dataset, rename_prediction_file
from evaluation import evaluate_results
import json
import os
import shutil
import subprocess


def delete_folder(name: str):
    try:
        shutil.rmtree(name)
        print("successfully deleted: %s" % name)
    except OSError as e:
        print("deleting failed: %s - %s." % (e.filename, e.strerror))


def check_docker_state():
    """checks if there is a running instance of docker.

    Uses the 'docker info' command to check if docker is running. If it raises an error,
    docker is either not installed or just not running.

    Raises:
        RuntimeError
    """
    check = subprocess.run("docker info", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode != 0:
        error_msg = "No running instance of 'docker' is found. It is required to build or run container."
        raise RuntimeError(error_msg)


def create_directories():
    # has to check if exist and if not create directories for the porject (archive, generated-tests, predictions...)
    pass


def run_docker_container(name: str, *bindings: list, option=None):
    """Starts a docker container.

    Args:
        name: Name of the docker to start.
        *bindings:
            List of string couples. Contains the mount bindings with the source relative
            to the current dictionary as the first string and the absolute target path as the second
        option: Optional argument which is passed to the docker.

    Returns:
    """
    check_docker_state()
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


def build_docker_collection(d_list: list):
    """Builds a set of docker images defined by a list of dictionaries.

    Args:
        d_list: list of dictionaries with data to set up docker.
    """
    check_docker_state()
    for docker in d_list:
        docker_folder = docker["folder"]
        docker_name = docker["name"]
        cmd = f'cmd /c "docker build -t {docker_name} {docker_folder}"'
        os.system(cmd)


def run_docker_collection(d_list: list):
    """Runs a set of docker container defined by a list of dictionaries.

    Args:
        d_list: list of dictionaries with data to set up docker.
    """
    for docker in d_list:
        docker_name = docker["name"]
        bindings = []
        for key in [*docker]:
            if "binding" in key:
                bindings.append(docker[key])
        option = None
        if "option" in [*docker]:
            option = docker["option"]
        print(f"run {docker_name}")
        run_docker_container(docker_name, *bindings, option=option)


def split_docker_list(d_list: list):
    """Splits the list of docker in docker marked as generator and docker without that mark (test environment docker)

    Args:
        d_list: list of dictionaries with docker information

    Returns:
        (list, list):
            - list of dictionaries with test generator docker information (can not contain more than one element)
            - list of dictionaries with test environment docker information
    """
    test_d_list = []
    generator_d_list = []
    num_test_generator = 0
    for docker in d_list:
        if "generator" in [*docker]:
            generator_d_list.append(docker)
            num_test_generator += 1
            if num_test_generator > 1:
                error_msg = "More than one docker is defined as 'generator'. " \
                            "Only one item in dockerlist json file should contain the 'generator' key."
                raise RuntimeError(error_msg)
        else:
            test_d_list.append(docker)
    return generator_d_list, test_d_list


def main(dockerlist_file: str):
    delete_folder("generated-tests")
    delete_folder("predictions")

    with open(dockerlist_file) as f:
        docker_list = json.load(f)

    build_docker_collection(docker_list)

    generator_docker_list, test_docker_list = split_docker_list(docker_list)

    run_docker_collection(generator_docker_list)

    overwrite_datasets = [["BreastCancer", "Zeroes"], ["BreastCancerMinMaxNorm", "VerySmall"],
                          ["BreastCancerMeanNorm", "Bias"], ["Wine", "LeftSkew"], ["WineMinMaxNorm", "RightSkew"],
                          ["WineMeanNorm", "Outlier"]]
    for overwrite_pair in overwrite_datasets:
        overwrite_dataset(overwrite_pair[0]+"_1_training.arff", overwrite_pair[1]+"_1_training.arff")
        overwrite_dataset(overwrite_pair[0]+"_1_test.arff", overwrite_pair[1]+"_1_test.arff")

    run_docker_collection(test_docker_list)

    for overwrite_pair in overwrite_datasets:
        rename_prediction_file(overwrite_pair[0], overwrite_pair[1])

    evaluate_results(print_all=False)


if __name__ == "__main__":
    main("dockerlist.json")
