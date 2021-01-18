"""System calls for the building of different docker container."""

import os

def build_my_docker():
	# stop all container, especially needed for the notebooks
	os.system('cmd /c "docker container stop $(docker container ls -aq)"')

	# run atoml docker build
	os.system('cmd /c "docker build -t atoml_docker atoml_docker"')

	# run sklearn docker build
	os.system('cmd /c "docker build -t sklearn_docker sklearn_docker"')

	# run weka docker build
	os.system('cmd /c "docker build -t weka_docker weka_docker"')

	# run spark docker build
	os.system('cmd /c "docker build -t spark_docker spark_docker"')

	# run caret docker build
	os.system('cmd /c "docker build -t caret_docker caret_docker"')

if __name__ == "__main__":
	build_my_docker()
