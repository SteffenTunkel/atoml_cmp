import os

# cmd /k -> remain ___ cmd /c -> terminate

# run atoml test generation
#os.system('cmd /c "docker run --mount type=bind,source=%cd%/generated-tests,target=/generated-tests --mount type=bind,source=%cd%/algorithm-descriptions,target=/testdata atoml_docker"')

# run sklearn classification
#os.system('cmd /c "docker run --mount type=bind,source=%cd%/predictions,target=/log --mount type=bind,source=%cd%/dataset,target=/data sklearn_docker"')

# run weka classification
#os.system('cmd /c "docker run --mount type=bind,source=%cd%/predictions,target=/log --mount type=bind,source=%cd%/dataset,target=/data weka_docker"')

# run spark classification
#os.system('cmd /c"docker run -p 8888:8888 --mount type=bind,source=%cd%/predictions,target=/home/jovyan/work/log --mount type=bind,source=%cd%/dataset,target=/data spark_docker"')

# run sklearn connected to atoml v1
#os.system('cmd /c "docker run --mount type=bind,source=%cd%/generated-tests/sklearn/smokedata,target=/smokedata --mount type=bind,source=%cd%/generated-tests/sklearn/morphdata,target=/morphdata --mount type=bind,source=%cd%/dataset,target=/data --mount type=bind,source=%cd%/predictions,target=/log sklearn_atoml_docker"')

os.system('cmd /c"docker run spark_docker"')


# run sklearn connected to atoml v2
#os.system('cmd /c "docker run --mount type=bind,source=%cd%/generated-tests/sklearn,target=/sklearn --mount type=bind,source=%cd%/predictions,target=/log sklearn_atoml_docker"')