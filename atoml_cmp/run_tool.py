"""Runs the tool."""

from atoml_cmp.external_data_utils import overwrite_dataset, rename_prediction_file
from atoml_cmp.evaluation import evaluate_results
import json
import os
import shutil
import subprocess
from pathlib import Path

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
    check = subprocess.run(["docker","info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode != 0:
        error_msg = "No running instance of 'docker' is found. It is required to build or run container."
        raise RuntimeError(error_msg)


def run_docker_container(name: str, *bindings: list, option=None):
    """Starts a docker container.

    Args:
        name: Name of the docker to start.
        *bindings:
            List of string couples. Contains the mount bindings with the source relative
            to the current dictionary as the first string and the absolute target path as the second
        option: Optional argument which is passed to the docker.

    Raises:
        RuntimeWarning: if the docker run failed.
    """
    check_docker_state()
    # created mounted folders on host if they don't exist
    for bind in bindings:
        Path(bind[0]).mkdir(parents=True, exist_ok=True)
    # build the command
    cmdlist = ["docker", "run"]
    for bind in bindings:
        cmdlist.append("--mount")
        bindstr = ""
        bindstr += 'type=bind,source='
        bindstr += os.path.join(os.getcwd(),bind[0])
        bindstr += ',target='
        bindstr += bind[1]
        cmdlist.append(bindstr)
    cmdlist.append(name)
    if option is not None:
        cmdlist.append(str(option))
    # run command
    cmd_return = subprocess.run(cmdlist)
    if cmd_return.returncode != 0:
        msg = f"Running container: {name} failed (returns {cmd_return.returncode})."
        # raise RuntimeWarning(msg)
        print(msg)


def build_docker_collection(d_list: list):
    """Builds a set of docker images defined by a list of dictionaries.

    Args:
        d_list: list of dictionaries with data to set up docker.

    Raises:
        RuntimeError: if a docker build failed.
    """
    check_docker_state()
    for docker in d_list:
        docker_folder = docker["folder"]
        docker_name = docker["name"]
        cmdlist = ["docker", "build", "-t", docker_name, docker_folder]
        cmd_return = subprocess.run(cmdlist)
        if cmd_return.returncode != 0:
            error_msg = f"Build of {docker_name} based on {docker_folder} failed."
            raise RuntimeError(error_msg)


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
    """Splits the list of docker in test generation docker and test environment docker.

    The split is done by checking for the 'generator' key in the dockers dict defined in the json file.
    Only the docker for the test case / test data generation should contain that key. All others, which are used to set
     the environment for running the tests, must not have it.

    Args:
        d_list: list of dictionaries with docker information

    Returns:
        (list, list):
            - list of dictionaries with test generator docker information (can not contain more than one element)
            - list of dictionaries with test environment docker information

    Raises:
        RuntimeError: if there is more than one test generator docker defined.
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


def main(dockerlist_file: str, gen_tests_folder="generated-tests", pred_folder="predictions",
         yaml_folder="algorithm-descriptions", archive_folder="archive"):
    delete_folder(gen_tests_folder)
    delete_folder(pred_folder)

    with open(dockerlist_file) as f:
        docker_list = json.load(f)

    build_docker_collection(docker_list)

    generator_docker_list, test_docker_list = split_docker_list(docker_list)

    run_docker_collection(generator_docker_list)

    overwrite_datasets = [["BreastCancer", "Zeroes"], ["BreastCancer-MinMaxNorm", "VerySmall"],
                          ["BreastCancer-ZNorm", "Bias"], ["Wine", "LeftSkew"], ["Wine-MinMaxNorm", "RightSkew"],
                          ["Wine-ZNorm", "Outlier"]]
    for overwrite_pair in overwrite_datasets:
        overwrite_dataset(overwrite_pair[0]+"_1_training.arff", overwrite_pair[1]+"_1_training.arff")
        overwrite_dataset(overwrite_pair[0]+"_1_test.arff", overwrite_pair[1]+"_1_test.arff")

    run_docker_collection(test_docker_list)

    for overwrite_pair in overwrite_datasets:
        rename_prediction_file(overwrite_pair[0], overwrite_pair[1])

    num_csv_files = evaluate_results(prediction_folder=pred_folder, yaml_folder=yaml_folder,
                                     archive_folder=archive_folder, print_all=False)
    return num_csv_files


if __name__ == "__main__":
    main("dockerlist.json")
