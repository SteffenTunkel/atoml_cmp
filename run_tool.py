'''This module runs the tool'''

import os
import sys
import shutil

import evaluation

# cmd /k -> remain ___ cmd /c -> terminate
# Example for a system call for a docker with a mount:
#os.system('cmd /c "docker run --mount type=bind,source=%cd%/algorithm-descriptions,target=/testdata atoml_docker"')

def runMyDocker(name, *bindings):
	'''
    Starts a docker container.
	The parameter name includes the container image name, which needs to be build before.
	The parameter bindings is a list of string couples. It contains the mount bindings with the source relative 
    to the current dictionary as the first string and the target as the second cmdstr = 'cmd /c"docker run '
    '''

    # build command string
    for bind in bindings:
        cmdstr += '--mount type=bind,source=%cd%'
        cmdstr += bind[0]
        cmdstr += ',target='
        cmdstr += bind[1]
        cmdstr += ' '
    cmdstr += name
    cmdstr += '"'
    # run command
    os.system(cmdstr)



def deleteFolder(name):
    try:
        shutil.rmtree(name)
        print("successfully deleted: %s" % name)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))


# clear generated-tests and predictions folder
deleteFolder("generated-tests")
deleteFolder("predictions")

# run atoml docker build
#os.system('cmd /c "docker build -t atoml_docker atoml_docker"')

atomlMounts = [["/generated-tests", "/generated-tests"],["/algorithm-descriptions", "/testdata"]]
runMyDocker("atoml_docker", *atomlMounts)
#times with gradle: 29.4, 26.2, 32.0
#times w/o gradle: 10-13s

sklearnMounts = [["/generated-tests/sklearn" , "/sklearn"], ["/predictions/sklearn", "/log"]]
runMyDocker("sklearn_docker", *sklearnMounts)

wekaMounts = [["/generated-tests/weka/src", "/code/src"], ["/predictions/weka", "/log"]]
runMyDocker("weka_docker", *wekaMounts)

sparkMounts = [["/generated-tests/spark/src", "/code/src"], ["/predictions/spark", "/log"]]
runMyDocker("spark_docker", *sparkMounts)
evaluation.jsonToCsvForSpark()


evaluation.evaluateResults()