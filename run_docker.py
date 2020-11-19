import os

# cmd /k -> remain ___ cmd /c -> terminate
# Example for a system call for a docker with a mount:
#os.system('cmd /c "docker run --mount type=bind,source=%cd%/algorithm-descriptions,target=/testdata atoml_docker"')

def runMyDocker(name, *bindings):
    cmdstr = 'cmd /c"docker run '
    for bind in bindings:
        cmdstr += '--mount type=bind,source=%cd%'
        cmdstr += bind[0]
        cmdstr += ',target='
        cmdstr += bind[1]
        cmdstr += ' '
    cmdstr += name
    cmdstr += '"'
    os.system(cmdstr)

atomlMounts = [["/generated-tests", "/generated-tests"],["/algorithm-descriptions", "/testdata"]]
runMyDocker("atoml_docker", *atomlMounts)

sparkMounts = [["/generated-tests/spark/src", "/code/src"], ["/predictions/spark", "/code/log"]]
runMyDocker("spark_docker", *sparkMounts)

wekaMounts = [["/generated-tests/weka/src", "/code/src"], ["/predictions/weka", "/code/log"]]
runMyDocker("weka_docker", *wekaMounts)

sklearnMounts = [["/generated-tests/sklearn" , "/sklearn"], ["/predictions/sklearn", "/log"]]
runMyDocker("sklearn_docker", *sklearnMounts)


