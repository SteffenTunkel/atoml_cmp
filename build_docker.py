import os

# stop all container, especially needed for the notebooks

#os.system('cmd /c "docker container stop $(docker container ls -aq)"')

# run atoml docker build
os.system('cmd /c "docker build -t atoml_docker atoml_docker"')

# run sklearn docker build
os.system('cmd /c "docker build -t sklearn_docker sklearn_docker"')

# run weka docker build
os.system('cmd /c "docker build -t weka_docker weka_docker"')

# run spark docker build
os.system('cmd /c "docker build -t spark_docker spark_docker"')



