FROM python:3.7

# set working directory
WORKDIR /container

# copy files/folder to build the system
COPY requirements.txt .

# install python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY run_atoml_sklearn_testsuite.py .

CMD ["python", "run_atoml_sklearn_testsuite.py"]